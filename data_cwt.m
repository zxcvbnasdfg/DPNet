clear;clc;close all;format compact

load microseism_combined.mat
N=1400;
fs=1000;
t=0:1/fs:N/fs;

%% 频图
wavename='cmor3-3';
totalscal=256;
Fc=centfrq(wavename);
c=2*Fc*totalscal;
scals=c./(1:totalscal);
f=scal2frq(scals,wavename,1/fs);

%%小波变换并保存成图像
[m,n]=size(train_X);
[~,label]=max(train_Y,[],2);
label=label-1;

fig = figure;
set(fig, 'Position', [0, 0, 800, 400]);

for i = 1537:1537
    % 计算小波系数
    coefs = cwt(train_X(i, :), scals, wavename);
    
    % 绘制小波图
    imagesc(t, f, abs(coefs) / max(max(abs(coefs))));
    
    % 添加坐标轴标签
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    
    % 保存矢量图
    filename_eps = ['E:/数据/小波/train_data/', num2str(i), '-', num2str(label(i)), '.eps'];
    print(filename_eps, '-depsc', '-r300');
    
    clf; % 清空图形以准备下一个循环
end




for i=1537:1540
    coefs=cwt(train_X(i,:),scals,wavename); 
    % img=abs(coefs)/max(max(abs(coefs)));
    imagesc(t,f,abs(coefs)/max(max(abs(coefs))));
    %imagesc(t,f,log(abs(coefs)));

    % 添加坐标轴标签
    xlabel('Time(s)');  
    ylabel('Frequency(Hz)'); 
    
    % set(gca,'position',[0 0 1 1])
    
    %fname=['E:/数据/小波/train_data/',num2str(i),'-',num2str(label(i)),'.jpg'];
    filename_eps = ['E:/数据/小波/train_data/', num2str(i), '-', num2str(label(i)), '.eps'];
    %saveas(gcf,fname);
    saveas(gcf, filename_eps, 'epsc');
    clf;
end

%{
[m,n]=size(valid_X);
[~,label]=max(valid_Y,[],2);
label=label-1;
for i=1:m
    i
    coefs=cwt(valid_X(i,:),scals,wavename); 
    %img=abs(coefs)/max(max(abs(coefs)));
    imagesc(t,f,abs(coefs)/max(max(abs(coefs))));  
    %imagesc(t,f,log(abs(coefs)));
    
    set(gca,'position',[0 0 1 1])
    fname=['E:/数据/小波/valid1_img/',num2str(i),'-',num2str(label(i)),'.jpg'];
    saveas(gcf,fname);
end

[m,n]=size(test_X);
[~,label]=max(test_Y,[],2);
label=label-1;
for i=1:m
    i
    coefs=cwt(test_X(i,:),scals,wavename); 
    % img=abs(coefs)/max(max(abs(coefs)));
    imagesc(t,f,abs(coefs)/max(max(abs(coefs))));
    %imagesc(t,f,log(abs(coefs)));
    set(gca,'position',[0 0 1 1])
    fname=['E:/数据/小波/test1_img/',num2str(i),'-',num2str(label(i)),'.jpg'];
    saveas(gcf,fname);
end
%}