clear;clc;close all;format compact

load microseism_combined.mat
N=1400;
fs=1000;
t=0:1/fs:N/fs;

%% Ƶͼ
wavename='cmor3-3';
totalscal=256;
Fc=centfrq(wavename);
c=2*Fc*totalscal;
scals=c./(1:totalscal);
f=scal2frq(scals,wavename,1/fs);

%%С���任�������ͼ��
[m,n]=size(train_X);
[~,label]=max(train_Y,[],2);
label=label-1;

fig = figure;
set(fig, 'Position', [0, 0, 800, 400]);

for i = 1537:1537
    % ����С��ϵ��
    coefs = cwt(train_X(i, :), scals, wavename);
    
    % ����С��ͼ
    imagesc(t, f, abs(coefs) / max(max(abs(coefs))));
    
    % ����������ǩ
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    
    % ����ʸ��ͼ
    filename_eps = ['E:/����/С��/train_data/', num2str(i), '-', num2str(label(i)), '.eps'];
    print(filename_eps, '-depsc', '-r300');
    
    clf; % ���ͼ����׼����һ��ѭ��
end




for i=1537:1540
    coefs=cwt(train_X(i,:),scals,wavename); 
    % img=abs(coefs)/max(max(abs(coefs)));
    imagesc(t,f,abs(coefs)/max(max(abs(coefs))));
    %imagesc(t,f,log(abs(coefs)));

    % ����������ǩ
    xlabel('Time(s)');  
    ylabel('Frequency(Hz)'); 
    
    % set(gca,'position',[0 0 1 1])
    
    %fname=['E:/����/С��/train_data/',num2str(i),'-',num2str(label(i)),'.jpg'];
    filename_eps = ['E:/����/С��/train_data/', num2str(i), '-', num2str(label(i)), '.eps'];
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
    fname=['E:/����/С��/valid1_img/',num2str(i),'-',num2str(label(i)),'.jpg'];
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
    fname=['E:/����/С��/test1_img/',num2str(i),'-',num2str(label(i)),'.jpg'];
    saveas(gcf,fname);
end
%}