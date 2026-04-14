clear
clc
close all
%% 读数据
save_path = 'D:\程序\Lff\检查122054';
shot= 85011;
time1=2;
time2=4;
fftpoint = 256;
mdsconnect('mds.ipp.ac.cn');
mdsopen('east',shot);
%% 2代
%21年探针改造前 fftpoint=128
%LIIS01-15；LOIS01-20
%UIIS01-28；UOIS01-26
%21年探针改造后
%LOIS01-32;%LIIS01-12

n = 13; % 定义探针个数
for i = 1:n
    signalname = ['\uois', num2str(i,'%02d')]; %常规磁诊断'\kmp l\h\u\d','\cmp', num2str(i),'t''n' 1-10
%     signalname = ['\KMPL', num2str(i),'N'];  %高频磁探针 LHP6-8T，KHP1-3、6-8T
%     在这里使用 signalname 来读取探头数据
    y = mdsvalue(['_sig=', signalname]);
    t = mdsvalue('dim_of(_sig)');
    index = find(t >= time1 & t <= time2); % 取值范围
    S_east = y(index);
    t_east = t(index);

    %% Spectrogram of the original signal
    ts = t_east(2) - t_east(1);
    fs = 1 / ts;
    g = 2 * fftpoint;
    [t_spec1, P1, f1] = spec_3D(t_east, S_east, fs, fftpoint, g);

    figure(1)
    imagesc(t_spec1, f1, log10(P1));
    colormap('jet');
    set(gca, 'YDir', 'normal')
%     set(gca,'CLim',[-8 -4])
%     ylim([0 50])
    xlabel('time (s)')
    ylabel('Frequency (kHz)')
    title(['#', num2str(shot), ' ', num2str(signalname(2:end))]); % 信号名
%     saveas(gcf, fullfile(save_path, sprintf('%s.fig', extractAfter(signalname, 1))));
%     saveas(gcf, fullfile(save_path, sprintf('%s.png', extractAfter(signalname, 1))));
    saveas(gcf, fullfile(save_path, sprintf('%s_%s.fig', num2str(shot), signalname(2:end))));
    saveas(gcf, fullfile(save_path, sprintf('%s_%s.png', num2str(shot), signalname(2:end))));
    close

end
    mdsdisconnect;
%% spec_3D & psd函数，画图用的！
function [t_spec,P,f]=spec_3D(t,signal,fs,fftpoint,g)
     L=length(signal);
     m=0;
     while (m/2+1)*g<L
           k=(m*g/2+1):(m/2+1)*g;
           y_temp=signal(k);
           y_temp=y_temp-mean(y_temp);
           m=m+1;
           t_spec(m)=mean(t(k));
          [P(:,m),f]=psd_me(y_temp,fs,fftpoint); 
     end
     f=f/10^3; 
end

function [P,f]=psd_me(x,fs,fftpoint)
%========================
%calculate the power spectrum density of signal x through
%        Pxx(f)=<x(f)*x(f)*/N>, <...> represents ensemble average
%fs is sampling frequency, fftpoint is fft point.
%x must be a vector. 
%the returned value P is Pxx(f), f is the frequency. 50% overlap
%========================
    if nargin~=3
    error('the standard form is [PSD,frequency]=psd_me(signal,sample_frequency,fftpoint)');
    else
    a=length(x);
    L=fix(a/fftpoint);
    m=2*L-1;
       for k=1:m
       j=(k-1)*fftpoint/2;
       p=(j+1):(j+fftpoint);
       y(1:fftpoint,k)=x(p);
       end
       y_fft=fft(y)/fftpoint;
       y_psd=abs(y_fft).^2;
       y_psd=mean(y_psd,2);
       P=fftshift(y_psd);
       f=[-fftpoint/2:(fftpoint/2-1)]*(fs/fftpoint);
    end

end