%%%
% Header
% fft_test
% Simulating data processing
% ADC and frequency analyser
%
%Francisco Costa
% SWE-2022
%%%

%%
clear all
close all
clc
%%


n_bits = 12; % N° bits ADC
V_ref  = 5; % reference voltaje ADC
% SINAD  = 69.9;
lidar_wavelength = 1550e-9;
fs_av          = 50e6; % sampling frequency
L           = 10^5;  %length of the signal. Cannot be less than n_fftpoints!!!
n_fftpoints = [ 2^7,2^7,2^8,2^9,2^10];   % n° of points for each block (fft points). Caannot be higher than L!!!
% Ts          = 1./fs; % Sampling period
signal_f    = 20e6;   % Frequency of the signal
level_noise = 5;

%% Uncertainty in the sampling frequency. 
% Gaussian distributed sampling frequency uncertainty with mean = fs_av and stdv = sigma_feq
N=5;
bias_fs_av = 7e6; % +/- 

lb = fs_av-bias_fs_av;
ub = fs_av+bias_fs_av;
% Rectangular random distribution of frequencies and periods. The
% probability is 1 for values within the range [lb,ub] and 0 outside.
fs = (ub-lb).*rand(N,1) + lb;
Ts = 1./fs;

% Same but with GUM
% std_fs_av = bias_fs_av/sqrt(3);
% fs = std_fs_av.*randn(N,1) + fs_av; 
% Ts = 1./fs;


%% Loop to calculate the frequency spectra for each fs

for ind_fs=1:length(fs)

    t{ind_fs}  = (0:L-1)*Ts(ind_fs); %t vector
    tf{ind_fs} = (0:n_fftpoints(1,ind_fs)-1)*Ts(ind_fs); %tf vector
    f{ind_fs}  = linspace(0,fs(ind_fs)/2,floor(length(t{ind_fs})/2+1));
    ff{ind_fs} = linspace(0,fs(ind_fs)/2,floor(length(tf{ind_fs})/2+1));
    %     ff{ind_L} = 0:fs/n_fftpoints:fs(ind_L)/2;
    % Signal:
    S{ind_fs}  = 10*sin(2*pi*signal_f.*t{ind_fs});%+ 3*sin(2*pi*abs(randn(1,1))*signal_f*t{ind_L})+sin(2*pi*abs(randn(1,1))*signal_f*t{ind_L})+ sin(2*pi*abs(randn(1,1))*signal_f*t{ind_L})+...
    % .74*sin(2*pi*signal_f.*t{ind_L})+ 1.7*sin(2*pi*abs(randn(1,1))*signal_f*t{ind_L})+sin(2*pi*abs(randn(1,1))*signal_f*t{ind_L})+ sin(2*pi*abs(randn(1,1))*signal_f*t{ind_L});% Adding up Signal contributors
    
    % introduce  normally distributed random noise in the signal
    noise=level_noise*randn(size(t{ind_fs}));
    stdv_n=std(noise);
    X0{ind_fs} = (S{ind_fs} + noise);
    X{ind_fs} =  X0{ind_fs}./max(abs(X0{ind_fs}))  ;
    
    %%%%%%%%%%%%%%%%%%
    % ADC:
    % Quantization of the signal
    ENOB = n_bits; %(SINAD-1.76)/6.02;
    vres= (2/(2^ENOB));
    % find upper and lower limits
    low_lim= sort(-vres:-vres: -1, 'ascend');
    upp_lim=(vres:vres:1);
    v=[low_lim,0,upp_lim]; % vector of ADC quantized values
    
    [~,mm,ttt]=histcounts(X{ind_fs},v); % calculates the indexes of the bins X values belong to
    F_mid = conv(v, [0.5,0.5], 'valid'); % calculates the intermediate value of the elements in v (the gaps' values)--> V values of quantization
    quant=F_mid(ttt);
    S_quant = quant;
    % Mean the quantize original signal (mean and get the closest to the
    % original value
    for ind_quant=1:size(S_quant,2)
        [s,b]=min(abs(S_quant(:,ind_quant)-mean(S_quant(:,ind_quant))));
        mean_S_quant{ind_fs}(:,ind_quant)= S_quant(b,ind_quant);
    end
    
    
    %%%%%%%%%%%%%%%%%%
    % Frequency analyser:    
    % FFT of the Quantised signal

    % 1st chop up the signal in "n_blocks" intervals:
    n_blocks = floor(L/n_fftpoints(1,ind_fs));
    n=1;
    nblocks(1,ind_fs)=n_blocks;
    for ind_chop=1:n_blocks
        %         S_quant_chop(ind_chop,:)=S_quant(1,n:ind_chop*n_fftpoints(1,ind_L));
        S_quant_chop(ind_chop,:)=mean_S_quant{ind_fs}(1,n:ind_chop*n_fftpoints(1,ind_fs));
        tic;
        
        % 2nd calculate the fft for each interval
        for ind_fft=1:1
            % FFT:
            P3 = fft(S_quant_chop(ind_chop,:)'); % Fourier transform
            P2= abs(P3')/n_fftpoints(1,ind_fs);
            P1 = P2(:,1:n_fftpoints(1,ind_fs)/2+1);
            %             P1(2:end-1) = 2*P1(2:end-1);
            S_fft_quant{ind_chop}(ind_fft,:)=P1.^2;           
        end
        
        T(1,ind_fs)=toc;
        n=n+n_fftpoints(1,ind_fs);
        
        % Accumulate results of the trimmed fft signal
        S_fft_quant_chop {ind_fs} (ind_chop,:) = mean (S_fft_quant{ind_chop},1);
    end
    
    % Time taken for each fft
    Time_fft(1,ind_fs)=sum(T);
    
    % Average all the intervals
    S_mean_fft_quant{ind_fs} =mean(S_fft_quant_chop {ind_fs},1);
    
    % Peak detection from the averaged spectra
    [ii_mean,ii1_mean]    = max(S_mean_fft_quant{ind_fs});
    f_peak(ind_fs)         = ff{ind_fs} (ii1_mean);

    % Assessing Statistics:    
    % mode_S_quant{ind_L}   = mode(S_quant);
    % mean_fft_quant{ind_L} = mean(S_fft_quant{ind_L},1);
    % mean_fft_quant{ind_L} = (S_fft_quant{ind_L});
    

    % Velocity resolution
    V_resolution(1,ind_fs)=0.5*lidar_wavelength*(2*f{ind_fs}(2));
    mean_S{ind_fs}         = mean(X{ind_fs},1);
    stdv_det{ind_fs}       = mean(stdv_n)/length(stdv_n); % experimental stdv
    
    % Relative error in the peak detection
    Relative_error(ind_fs,:) =  100*abs((signal_f - f_peak(ind_fs)))/signal_f;
    stdv_signal (ind_fs,:) = level_noise/sqrt(size(P3,1));
    % STDV_signal (ind_L,:) = level_noise/sqrt(size(P3,1));
    RMSE{ind_fs} =sqrt(mean((mean_S{ind_fs}-mean_S_quant{ind_fs}).^2));
    SNQR{ind_fs} =((6.02*n_bits-1.25));
    
    % Clear variables
    clear P3
    clear S_quant
    clear S_quant_chop
    clear S_fft_quant
    
end
format shortEng
mean_freq=mean(f_peak);
stdv_freq=std(f_peak);
disp(['Peak frequency mean: ',num2str(mean_freq)])
disp(['Peak frequency variance: ',num2str((stdv_freq)^2)])
format short
%% PLOTS

% Plotting signals:
markersize=[{'k-*'},{'b-*'},{'r-*'},{'g-*'},{'c-*'},{'m-*'},{'y-*'}];
markersize_2=[{'r--x'},{'b--x'},{'r--x'},{'g--x'},{'c--x'},{'m--x'},{'y--x'}];
markersize_3=[{'k-.'},{'b-.'},{'r-.'},{'g-.'},{'c-.'},{'m-.'},{'y-.'}];
figure,hold on
for in_sig=1:length(fs)
    Legend1{in_sig}=['fs = ',num2str(fs(in_sig),'%.2s') '; N°points fft = ',num2str(n_fftpoints(1,in_sig)), '; SNQR = ', num2str(SNQR{in_sig},'%.2s'),' dB'];
    plot(t{in_sig}(1,:)  ,mean_S{in_sig}(1,:),markersize{in_sig},'Linewidth',1.2,'displayname',Legend1{in_sig})
    plot(t{in_sig},mean_S_quant{in_sig},markersize_2{in_sig},'Linewidth',1.9,'displayname','Q');
    plot(t{in_sig},mean_S{in_sig}(1,:)-mean_S_quant{in_sig},markersize_3{in_sig},'Linewidth',1.9,'displayname',['Quantization error; N° bits =',num2str(n_bits) ]);
    %     plot(t{i} ,X{i}(1,:)  ,markersize{i})
    %     plot(t{i},S_quant0{i},markersize_2{i},'Linewidth',1.4);
end
xlabel('time [s]', 'fontsize',20)
ylabel('[-]', 'fontsize',20)
title(['Signal -',' sigma = ',num2str(level_noise),'; n°bits = ', num2str(n_bits)],'fontsize',25)
hold off
legend show
grid on
set(gca,'FontSize',20);


% % fft signals
figure, hold on
for in_fft=1:length(fs)
    Legend2{in_fft}=['fs = ',num2str(fs(in_fft),'%.2s'), '; L = ',num2str(L), '; N°points fft = ',num2str(n_fftpoints(1,in_fft)),'; RE[%] = ', num2str(round(Relative_error(in_fft,:),2)),'; n°bits = ', num2str(n_bits) ];
    plot(ff{in_fft},S_mean_fft_quant{in_fft},markersize{in_fft},'displayname',Legend2{in_fft});
end
hold off
legend
title('Frequency spectra', 'fontsize',20)
xlabel('f [Hz]', 'fontsize',20)
ylabel('[-]', 'fontsize',20)
grid on
% l1.MarkerFaceColor = l1.Color;
set(gca,'FontSize',20);


% Plotting sensitivity annalysis
figure,plot(n_fftpoints,Relative_error,'bo')
title('Relative error vs. N°of samples', 'fontsize',20)
ylabel('Relative error [%]', 'fontsize',20)
xlabel('Number of fft points', 'fontsize',20)
grid on
figure,plot(n_fftpoints,T,'bo')
title('Computational time vs. N°of samples', 'fontsize',20)
ylabel('time[s]', 'fontsize',20)
xlabel('Number of fft points', 'fontsize',20)
legend
grid on
set(gca,'FontSize',20);% %
% figure,plot(fs,Relative_error,'bo')
% title('Relative error vs. Sampling frequency', 'fontsize',20)
% ylabel('Relative error [%]', 'fontsize',20)
% xlabel('Sampling frequency [Hz]', 'fontsize',20)



% t_inf = 0:.001:2000;
% t_inf2 = 0:.01:2000;
% Signal = 5*sin(2*pi*signal_f.*t_inf)+ .9*sin(2*pi*abs(randn(1,1))*signal_f*t_inf)+sin(2*pi*abs(randn(1,1))*signal_f*t_inf)+ sin(2*pi*abs(randn(1,1))*signal_f*t_inf)+...
%     1.1*sin(2*pi*signal_f.*t_inf)+ .5*sin(2*pi*abs(randn(1,1))*signal_f*t_inf)+sin(2*pi*abs(randn(1,1))*signal_f*t_inf)+ sin(2*pi*abs(randn(1,1))*signal_f*t_inf);
% Signal2 = 5*sin(2*pi*signal_f.*t_inf2)+ .9*sin(2*pi*abs(randn(1,1))*signal_f*t_inf2)+sin(2*pi*abs(randn(1,1))*signal_f*t_inf2)+ sin(2*pi*abs(randn(1,1))*signal_f*t_inf2)+...
%     1.1*sin(2*pi*signal_f.*t_inf2)+ .5*sin(2*pi*abs(randn(1,1))*signal_f*t_inf2)+sin(2*pi*abs(randn(1,1))*signal_f*t_inf2)+ sin(2*pi*abs(randn(1,1))*signal_f*t_inf2);
