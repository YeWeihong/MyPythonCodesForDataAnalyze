import argparse
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


CROSSREF_API = "https://api.crossref.org/works"
USER_AGENT = "iop-keyword-downloader/1.0 (mailto:your_email@example.com)"
DEFAULT_TIMEOUT = 60


# =========================
# 数据库
# =========================
def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            doi TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT,
            year INTEGER,
            journal TEXT,
            keyword TEXT,
            landing_url TEXT,
            pdf_url TEXT,
            pdf_path TEXT,
            status TEXT NOT NULL DEFAULT 'new',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            downloaded_at TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_keyword ON papers(keyword)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status)")
    conn.commit()
    return conn


def paper_exists(conn: sqlite3.Connection, doi: str) -> bool:
    cur = conn.execute("SELECT 1 FROM papers WHERE doi = ?", (doi.lower(),))
    return cur.fetchone() is not None


def upsert_paper(conn: sqlite3.Connection, paper: Dict) -> None:
    conn.execute("""
        INSERT INTO papers
        (doi, title, authors, year, journal, keyword, landing_url, pdf_url, pdf_path, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doi) DO UPDATE SET
            title=excluded.title,
            authors=excluded.authors,
            year=excluded.year,
            journal=excluded.journal,
            keyword=CASE
                WHEN papers.keyword IS NULL OR papers.keyword = '' THEN excluded.keyword
                ELSE papers.keyword
            END,
            landing_url=excluded.landing_url
    """, (
        paper["doi"],
        paper["title"],
        paper["authors"],
        paper["year"],
        paper["journal"],
        paper["keyword"],
        paper["landing_url"],
        paper.get("pdf_url", ""),
        paper.get("pdf_path", ""),
        paper.get("status", "new"),
    ))
    conn.commit()


def mark_downloaded(conn: sqlite3.Connection, doi: str, pdf_url: str, pdf_path: str) -> None:
    conn.execute("""
        UPDATE papers
        SET pdf_url = ?, pdf_path = ?, status = 'downloaded', downloaded_at = ?
        WHERE doi = ?
    """, (pdf_url, pdf_path, datetime.now().isoformat(timespec="seconds"), doi.lower()))
    conn.commit()


def mark_failed(conn: sqlite3.Connection, doi: str, pdf_url: str = "") -> None:
    conn.execute("""
        UPDATE papers
        SET pdf_url = ?, status = 'failed'
        WHERE doi = ?
    """, (pdf_url, doi.lower()))
    conn.commit()


# =========================
# 工具函数
# =========================
def normalize_doi(doi: str) -> str:
    doi = doi.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return doi.lower()


def sanitize_filename(name: str, max_len: int = 180) -> str:
    name = re.sub(r'[\\/*?:"<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:max_len]


def parse_year(item: Dict) -> Optional[int]:
    for key in ["published-print", "published-online", "issued", "created"]:
        part = item.get(key)
        if not part:
            continue
        date_parts = part.get("date-parts", [])
        if date_parts and date_parts[0]:
            try:
                return int(date_parts[0][0])
            except Exception:
                pass
    return None


def parse_authors(item: Dict) -> str:
    authors = []
    for a in item.get("author", []):
        given = a.get("given", "").strip()
        family = a.get("family", "").strip()
        name = " ".join([x for x in [given, family] if x]).strip()
        if name:
            authors.append(name)
    return "; ".join(authors)


# =========================
# Crossref 检索
# =========================
def crossref_search(
    keyword: str,
    journal: str,
    max_results: int = 100,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> List[Dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    filters = [
        "type:journal-article",
        f"container-title:{journal}",
    ]
    if since:
        filters.append(f"from-pub-date:{since}")
    if until:
        filters.append(f"until-pub-date:{until}")

    papers = []
    cursor = "*"

    while len(papers) < max_results:
        rows = min(100, max_results - len(papers))
        params = {
            "query": keyword,
            "filter": ",".join(filters),
            "rows": rows,
            "cursor": cursor,
            "select": "DOI,title,author,issued,published-print,published-online,container-title,URL",
        }

        resp = session.get(CROSSREF_API, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()["message"]

        items = data.get("items", [])
        if not items:
            break

        for item in items:
            doi = item.get("DOI")
            title_list = item.get("title", [])
            journal_list = item.get("container-title", [])
            if not doi or not title_list:
                continue

            papers.append({
                "doi": normalize_doi(doi),
                "title": title_list[0].strip(),
                "authors": parse_authors(item),
                "year": parse_year(item),
                "journal": journal_list[0].strip() if journal_list else journal,
                "landing_url": f"https://doi.org/{normalize_doi(doi)}",
            })

        next_cursor = data.get("next-cursor")
        if not next_cursor or next_cursor == cursor:
            break

        cursor = next_cursor
        time.sleep(1.0)

    # DOI 去重
    seen = set()
    deduped = []
    for p in papers:
        if p["doi"] in seen:
            continue
        seen.add(p["doi"])
        deduped.append(p)

    return deduped


# =========================
# 浏览器
# =========================
def build_driver(
    download_dir: Path,
    chrome_user_data_dir: Optional[str] = None,
    chrome_profile: Optional[str] = None,
    headless: bool = False,
) -> webdriver.Chrome:
    options = Options()
    options.add_argument("--start-maximized")
    options.add_experimental_option("prefs", {
        "download.default_directory": str(download_dir.resolve()),
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True,
    })

    if chrome_user_data_dir:
        options.add_argument(f"--user-data-dir={chrome_user_data_dir}")
    if chrome_profile:
        options.add_argument(f"--profile-directory={chrome_profile}")
    if headless:
        options.add_argument("--headless=new")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(DEFAULT_TIMEOUT)
    return driver


def wait_for_user_login(driver: webdriver.Chrome) -> None:
    driver.get("https://iopscience.iop.org/")
    print("\n请在打开的浏览器中确认你已经具有 IOP 全文访问权限。")
    print("若需要登录代理/校园认证，请手动完成。完成后回车继续。")
    input("按回车继续...")


# =========================
# PDF 解析与下载
# =========================
def extract_pdf_url_from_page(driver: webdriver.Chrome) -> str:
    # 先尝试通用 meta 标签
    meta_url = driver.execute_script("""
        const selectors = [
            'meta[name="citation_pdf_url"]',
            'meta[property="citation_pdf_url"]'
        ];
        for (const s of selectors) {
            const el = document.querySelector(s);
            if (el && el.content) return el.content;
        }
        return "";
    """)
    if meta_url:
        return meta_url

    # 再扫页面里的链接
    candidates = []
    anchors = driver.find_elements(By.TAG_NAME, "a")
    for a in anchors:
        href = (a.get_attribute("href") or "").strip()
        text = (a.text or "").strip().lower()
        if not href:
            continue

        href_l = href.lower()
        score = 0

        if href_l.endswith(".pdf") or ".pdf?" in href_l:
            score += 100
        if "/pdf" in href_l:
            score += 80
        if "article/pdf" in href_l:
            score += 60
        if "pdf" in text:
            score += 20
        if "download" in text:
            score += 10
        if "iopscience" in href_l:
            score += 5

        if score > 0:
            candidates.append((score, href))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def get_pdf_url_via_doi(driver: webdriver.Chrome, doi: str) -> str:
    landing = f"https://doi.org/{doi}"
    driver.get(landing)
    time.sleep(3)

    pdf_url = extract_pdf_url_from_page(driver)
    if pdf_url:
        return urljoin(driver.current_url, pdf_url)

    return ""


def selenium_cookies_to_requests(driver: webdriver.Chrome) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for c in driver.get_cookies():
        session.cookies.set(
            c["name"],
            c["value"],
            domain=c.get("domain"),
            path=c.get("path", "/"),
        )
    return session


def download_pdf_with_session(session: requests.Session, pdf_url: str, out_path: Path) -> bool:
    try:
        with session.get(pdf_url, stream=True, allow_redirects=True, timeout=DEFAULT_TIMEOUT) as r:
            r.raise_for_status()

            first_chunk = b""
            iterator = r.iter_content(chunk_size=8192)
            try:
                first_chunk = next(iterator)
            except StopIteration:
                return False

            content_type = (r.headers.get("Content-Type") or "").lower()
            is_pdf = first_chunk.startswith(b"%PDF") or ("pdf" in content_type)

            if not is_pdf:
                return False

            with open(out_path, "wb") as f:
                f.write(first_chunk)
                for chunk in iterator:
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False


# =========================
# 主流程
# =========================
def main():
    parser = argparse.ArgumentParser(description="按关键词下载 IOP 期刊论文，并建立本地数据库")
    parser.add_argument("--keyword", required=True, help="检索关键词")
    parser.add_argument("--journals", default="Nuclear Fusion", help="逗号分隔的期刊名")
    parser.add_argument("--max-results", type=int, default=50, help="每个期刊最多抓多少条")
    parser.add_argument("--since", default=None, help="起始日期，如 2018-01-01")
    parser.add_argument("--until", default=None, help="结束日期，如 2026-12-31")
    parser.add_argument("--db", default="iop_papers.sqlite", help="SQLite 数据库文件")
    parser.add_argument("--out-dir", default="iop_pdfs", help="PDF 保存目录")
    parser.add_argument("--chrome-user-data-dir", default=None, help="Chrome 用户目录")
    parser.add_argument("--chrome-profile", default=None, help="Chrome profile 名，例如 Default")
    parser.add_argument("--headless", action="store_true", help="无头模式，首次调试不建议开")
    parser.add_argument("--search-only", action="store_true", help="只检索入库，不下载 PDF")
    args = parser.parse_args()

    journals = [j.strip() for j in args.journals.split(",") if j.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = init_db(Path(args.db))

    # 1) 检索并写入数据库
    total_new = 0
    total_found = 0

    for journal in journals:
        print(f"\n[检索] 期刊: {journal}")
        papers = crossref_search(
            keyword=args.keyword,
            journal=journal,
            max_results=args.max_results,
            since=args.since,
            until=args.until,
        )
        print(f"[检索] 命中 {len(papers)} 条")

        for p in papers:
            total_found += 1
            p["keyword"] = args.keyword
            p["status"] = "new"

            if paper_exists(conn, p["doi"]):
                print(f"  [跳过-已存在] {p['doi']}")
                continue

            upsert_paper(conn, p)
            total_new += 1
            print(f"  [入库] {p['year']} | {p['title']}")

    print(f"\n[统计] 本次检索总命中: {total_found}，新增入库: {total_new}")

    if args.search_only:
        print("[结束] 仅检索模式，不执行下载。")
        return

    # 2) 浏览器登录 / 下载
    driver = build_driver(
        download_dir=out_dir,
        chrome_user_data_dir=args.chrome_user_data_dir,
        chrome_profile=args.chrome_profile,
        headless=args.headless,
    )

    try:
        wait_for_user_login(driver)

        cur = conn.execute("""
            SELECT doi, title, year
            FROM papers
            WHERE status IN ('new', 'failed')
            ORDER BY year DESC
        """)
        rows = cur.fetchall()

        print(f"\n[下载] 待处理 {len(rows)} 条")
        for i, (doi, title, year) in enumerate(rows, start=1):
            print(f"\n[{i}/{len(rows)}] {doi}")
            print(f"标题: {title}")

            try:
                pdf_url = get_pdf_url_via_doi(driver, doi)
                if not pdf_url:
                    print("  [失败] 没找到 PDF 链接")
                    mark_failed(conn, doi, "")
                    continue

                session = selenium_cookies_to_requests(driver)

                file_name = sanitize_filename(f"{year or 'unknown'} - {title} [{doi.replace('/', '_')}].pdf")
                out_path = out_dir / file_name

                # 已存在文件则直接标记成功
                if out_path.exists() and out_path.stat().st_size > 0:
                    print("  [跳过] 文件已存在")
                    mark_downloaded(conn, doi, pdf_url, str(out_path.resolve()))
                    continue

                ok = download_pdf_with_session(session, pdf_url, out_path)
                if ok:
                    print(f"  [成功] {out_path.name}")
                    mark_downloaded(conn, doi, pdf_url, str(out_path.resolve()))
                else:
                    print("  [失败] 下载返回的不是 PDF，或权限不足")
                    mark_failed(conn, doi, pdf_url)

                time.sleep(1.5)

            except Exception as e:
                print(f"  [异常] {e}")
                mark_failed(conn, doi, "")

    finally:
        driver.quit()
        conn.close()

    print("\n[完成] 全部流程结束。")


if __name__ == "__main__":
    main()