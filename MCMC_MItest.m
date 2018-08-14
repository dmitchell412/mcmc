
%init
load MI_QALAS_objfun_kernel_input.mat;
sampleSize=100;
dim=9;
i=1; j=1;
eta=zeros(1, dim);
acceptance=0;

ksx=77;
ksy=92;

ksrtemp=0;
ksitemp=0;

%prior mean and sd
pmu=eta_prior(:, 1)';
psigma=eta_prior(:, 2)';
%target mean/sd
tmu=eta_target(:, 1);
tsigma=eta_target(:, 2);

samples=zeros(sampleSize, dim);
MItemp=zeros(1, sampleSize);

%{
%to store vals causing -Inf
infeta=zeros(100, 9);
infksr=zeros(5, 151, 181);
infksi=zeros(5, 151, 181);
%}

while i<sampleSize
    %increment
    i=i+1;
    
        %sample from proposal probability distribution
        sample=mvnrnd(samples(i-1, :), psigma);
        [samples(i,:), ksr, ksi, acceptance]=metHas(sample, samples(i-1,:), pmu, psigma, acceptance, ksx, ksy);
    
    %set positive eta values
    eta=abs(samples(i, :));
    %calc and sum ksr and ksi
%     [ksr, ksi]=MI_QALAS_objfun_kernel(eta);
    pdfsamples(i)=ksr*ksi;%ksr(1,ksx,ksy)*ksi(1,ksx,ksy);
%     ksrtemp=ksrtemp+ksr;
%     ksitemp=ksitemp+ksi;
    
%     [MI, ~]=calcMI(ksrtemp, ksitemp, i);
%     MItemp(i)=MI;
    
    %{
    if isinf(MI)
        infksr=ksrtemp;
        infksi=ksitemp;
        infeta(j, :)=eta;
        j=j+1;
        break;
    end
    %}
end

% [MI, MIimg]=calcMI(ksrtemp, ksitemp, sampleSize);

%display img
%figure;
%imagesc(squeeze(MIimg(1, :, :)));

%plot MI at each sampleSize
try
    xxhat = -cumsum(log(abs(pdfsamples)))./(1:sampleSize);
figure;
analyticVal=1.1671e+06;
plot(1:sampleSize, MItemp);
hold on;
plot([0 sampleSize], [analyticVal analyticVal])
hold off;
title(strcat('MI: (', num2str(MI), ') vs Sample Size: (', num2str(sampleSize), ')'));
ylabel('MI');
xlabel('Sample Size');
legend({strcat('AV: ', num2str(analyticVal))}, 'FontSize', 12, 'TextColor', 'blue')
catch
end
%display histogram of samples
%{
figure;
for i=1:dim
    subplot(4, 4, i);
    %leftover zeros from init***
    hist(samples(:, i), 20);
    title(strcat('tmu: ', num2str(tmu(i)), ' tsigma: ', num2str(tsigma(i))));
    xlabel(strcat('std: ', num2str(std(samples(:, i))), ' mean: ', num2str(mean(samples(:, i)))));
end
%}

%Metropolis-Hastings Algo
function [val, tpdfsr, tpdfsi, acceptance]=metHas(sample, prev, pmu, psigma, acceptance, ksx, ksy)
    %target distribution
%     tpdf=@(x) normpdf(x, mu, sigma);
    [tpdfsr, tpdfsi]=MI_QALAS_objfun_kernel(sample);
    tpdfsr=squeeze(tpdfsr(1,ksx,ksy));
    tpdfsi=squeeze(tpdfsi(1,ksx,ksy));
    [tpdfpr, tpdfpi]=MI_QALAS_objfun_kernel(prev);
    tpdfpr=squeeze(tpdfpr(1,ksx,ksy));
    tpdfpi=squeeze(tpdfpi(1,ksx,ksy));
    %proposal distribution
    ppdf=@(x, y) mvnpdf(x, y);
    %draw random number
    r=rand();
    %corrective ratio
    c=ppdf(prev, sample)/ppdf(sample, prev);
    %determine acceptance
    alpha=min([1, (tpdfsr*tpdfsi)/(tpdfpr*tpdfpi)*c]);
    
    if (r<=alpha)
        %accept proposal
        val=sample;
        acceptance=acceptance+1;
    else
        %decline proposal
        val=prev;
    end
end

%calculate MI/MIimg
% function [MI, MIimg]=calcMI(ksr, ksi, sampleSize)
% load MI_QALAS_objfun_kernel_input.mat; %#ok<LOAD>
% Er=ksr/sampleSize;
% Ei=ksi/sampleSize;
% Sr=(ksr.^2)/sampleSize;
% Si=(ksi.^2)/sampleSize;
% Sri=(ksr.*ksi)/sampleSize;
% 
% signu=3.4762E-4;
% N=6;
% detSigz=(pi^(-N/2)*signu^2+Sr-Er.^2).*(pi^(-N/2)*signu^2+Si-Ei.^2)-(Sri-Er.*Ei).^2;
% Hz=0.5.*log((2*pi*2.7183)^2.*detSigz);
% Hzmu=0.5.*log((2*pi*2.7183)^2.*signu.^4);
% MIimg=Hz-Hzmu;
% 
% szmi=size(MIimg);
% pdv=1;
% szmi(2:1+ndims(subsmplmask{pdv}))=1; %#ok<NODEF>
% subsmplmask{pdv}=permute(subsmplmask{pdv},[ndims(subsmplmask{pdv})+1,1:ndims(subsmplmask{pdv})]);
% subsmplmask{pdv}=repmat(subsmplmask{pdv},szmi);
% MI=sum(MIimg(:).*subsmplmask{pdv}(:));
% 
% end