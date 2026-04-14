% 假设 x, y 已存在且为列向量
x0=get(gco,'xdata');y0=get(gco,'ydata');
x = x0(:);
y0 = y0(:);
y = smooth(y0(:),3);
N = numel(y);
ind=2;

% 1) 计算滑动窗口（前 2 后 2 共 5 点）的均值和标准差，ece用[100 100]，共201个点
if ind==1;
y_mean = movmean(y, [2 2]);   % 对每个位置，窗口 = 当前±2
y_std  = movstd( y0, [2 2]);   % 同理
elseif ind==2;
y_mean = movmean(y, [100 100]);   % 对每个位置，窗口 = 当前±2
y_std  = movstd( y0, [100 100]);   % 同理
end
% 2) 绘图
figure; hold on;

% 2.1) 标准差阴影区
Xfill = [x; flipud(x)]; 
Yfill = [y_mean + y_std; flipud(y_mean - y_std)];
fill(Xfill, Yfill, 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');

% 2.2) 滑动窗口均值曲线
plot(x, y_mean, 'r-', 'LineWidth', 2);

% 3) 美化
xlabel('时间 (x)');
ylabel('y');
title('y 随时间演化的滑动窗口均值和标准差 (\pm2 点窗口)');
legend('\mu \pm \sigma','\mu','Location','best');
grid on;
hold off;
