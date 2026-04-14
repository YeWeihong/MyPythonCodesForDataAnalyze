import requests

url = "https://news.inewsweek.cn/finance/"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
}

r = requests.get(url, headers=headers, timeout=15)

# ── 关键：用字节自己解码 ────────────────────────────────
raw_bytes = r.content

# 方案 A：优先尝试 GB18030（最兼容老中文站，包括 GB2312）
try:
    text = raw_bytes.decode('gb18030')
    print("使用 GB18030 解码成功")
except UnicodeDecodeError:
    # 失败再 fallback 到 utf-8
    text = raw_bytes.decode('utf-8', errors='replace')
    print("GB18030 失败，fallback 到 UTF-8（带替换字符）")

# ── 或者直接两种都试，哪个看起来正常用哪个 ──────────────
print("\n前 1500 字符（GB18030 解码）：")
print(raw_bytes.decode('gb18030', errors='replace')[:1500])

print("\n前 1500 字符（UTF-8 解码）：")
print(raw_bytes.decode('utf-8', errors='replace')[:1500])

# 如果想自动判断哪个更好，可以简单看 title 是否有大量 �
title_gb = raw_bytes.decode('gb18030', errors='replace').split('<title>')[1].split('</title>')[0]
title_utf8 = raw_bytes.decode('utf-8', errors='replace').split('<title>')[1].split('</title>')[0]

print("\nTitle (GB18030):", title_gb)
print("Title (UTF-8)  :", title_utf8)