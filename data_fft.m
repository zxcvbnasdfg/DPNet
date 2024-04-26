clear;clc;close all;format compact

load microseism_combined.mat
N=1400;
fs=1000;
t=0:1/fs:N/fs;

%进行FFT变换
[m,n]=size(train_X);
tz1=[];
for i=1:m
    i
    s=train_X(i,:);
    Y = fft(s);
    P2 = abs(Y/N);
    P1 = P2(1:N/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    tz1=[tz1;P1];
end
%验证集
[m,n]=size(valid_X);
tz2=[];
for i=1:m
    i
    s=valid_X(i,:);
    Y = fft(s);
    P2 = abs(Y/N);
    P1 = P2(1:N/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    tz2=[tz2;P1];
end
%% 测试集
[m,n]=size(test_X);
tz3=[];
for i=1:m
    i
    s=test_X(i,:);
    Y = fft(s);
    P2 = abs(Y/N);
    P1 = P2(1:N/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    tz3=[tz3;P1];
end
train_X=tz1;
valid_X=tz2;
test_X=tz3;

save FFT train_X valid_X test_X train_Y valid_Y test_Y