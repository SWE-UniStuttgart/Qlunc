%%%
% Header
% fft_test
% Simulating data processing
% ADC and frequency analyser

% Through Montecarlo simulations calculates:
%   - Uncertainty in vlos due to sampling frequency bias
%   - Uncertainty in vlos due to laser wavelength drift
%   - Uncertainty in vlos due to spectra average when more than 1 pulse is
%     selected
% Takes a signal discretised at fs and digitises it. A Fourier
% transform is applied to the digitised signal to obtain the frequency
% spectra. 
% Francisco Costa
% SWE-2022
%%%

%%
clear all %#ok<CLALL>
% close all
clc
format shortEng
%% Inputs

% distance         = 2000;
% PRF              = physconst('LightSpeed')/(2*distance);
n_bits           = 12;      % N_MC° bits ADC
V_ref            = 15;      % Reference voltaje ADC
lidar_wavelength = 1550e-9; % wavelength of the laser source.
fs_av            = 50e6;    % sampling frequency
L                = 2^10;    %length of the signal.
n_fftpoints      = L;       % n° of points for each block (fft points).
fd               = 2*V_ref/lidar_wavelength;  % Doppler frequency corresponding to Vref
level_noise      = 1.0; % noise as stdv added to the signal. Hardware comp before the signal downmixing
n_pulses         = 1;   % n pulses for averaging the spectra
N_MC             = 1e3; % n° MC samples to calculate the uncertainty due to bias in sampling frequency and wavelength

%% Uncertainty in the signal processing.

%%% Uncertainty in the sampling frequency %%%
bias_fs_av  = 2.5e6; % +/-
std_fs_av   = bias_fs_av/sqrt(3);
fs          = [fs_av;std_fs_av.*randn(N_MC,1) + fs_av];
Ts          = 1./fs;

% Accepted values
Tv = 1/fs_av;
tv = (0:n_fftpoints-1)*Tv;
fv = linspace(0,fs_av/2,floor(length(tv)/2+1));

%%% Stdv due drift in the wavelength of the laser %%%
stdv_wavelength  = .1e-9; % m

% Noisy wavelength vector:
noise_wavelength = stdv_wavelength;
wavelength_noise = lidar_wavelength+noise_wavelength*randn(N_MC+1,1);


%% Loop to calculate the frequency spectra for each fs
tic
for ind_npulses = 1:n_pulses   
    for ind_fs = 1:N_MC+1
        % Time and frequency vectors
        t{ind_fs} = (0:n_fftpoints-1)*Ts(ind_fs); %#ok<SAGROW> %time vector
        f{ind_fs}  = linspace(0,fs(ind_fs)/2,floor(length(t{ind_fs})/2+1)); %#ok<SAGROW> % floor()"/2+1" is added to match the length of the double-sided spectrum (P1)
        
        % Signal + Hardware noise:
        noise      = level_noise*randn(size(t{ind_fs}));
        S{ind_fs}  = noise+(10*sin(2*pi*fd.*t{ind_fs}) - 2.1*sin(2*pi*1.9*abs(randn(1,1))*fd*t{ind_fs}) + sin(2*pi*3*abs(randn(1,1))*fd*t{ind_fs})+...
                            1.24*sin(2*pi*6*abs(randn(1,1))*fd.*t{ind_fs}) + 1.7*sin(2*pi*2*abs(randn(1,1))*fd*t{ind_fs}) - 1.4*sin(2*pi*abs(randn(1,1))*fd*t{ind_fs}));%#ok<SAGROW> % Adding up Signal contributors
                
        % Spectrum function from matlab:
        [pxx{ind_fs},fr{ind_fs}] = pspectrum(S{ind_fs}./max(abs(S{ind_fs}))); %#ok<SAGROW>
        
        % Normalise signal:
        X{ind_fs} =  S{ind_fs}./max(abs(S{ind_fs}))  ; %#ok<SAGROW>
          
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
            mean_S_quant{ind_fs}(:,ind_quant)= S_quant(b,ind_quant); %#ok<SAGROW>
        end
        
        % RMSE due to digitisation process. If the signal is unbiased, RMSE
        % is the standard deviation.
        RMSE_digit(ind_fs,ind_npulses)= real(sqrt(sum((X{ind_fs}).^2-S_quant.^2)/length(t{ind_fs}))); %#ok<SAGROW>
        
        
        % tic
        %%%%%%%%%%%%%%%%%%
        % Frequency analyser:
        % FFT of the Quantised signal
        P3            = fft(mean_S_quant{ind_fs}'); % Fourier transform
        P2            = abs(P3')/n_fftpoints;
        P1            = P2(:,1:n_fftpoints/2+1);
        P1(2:end-1)   = 2*P1(2:end-1);
        S_fft_quant   = P1.^2;
        sss(ind_fs,:) = P1.^2; %#ok<SAGROW>
        
        % T(1,ind_fs)=toc;
        % Time_fft(1,ind_fs)=sum(T); % Time taken for each fft
        
        % Peak detection from the spectra
        [ii_mean,ii1_mean]          = max(S_fft_quant);
        f_peak(ind_fs,ind_npulses)  = f{ind_fs} (ii1_mean); %#ok<SAGROW>
        
        % Vlos
        v_MC(ind_fs,ind_npulses) = 0.5*wavelength_noise(ind_fs)*f_peak(ind_fs,ind_npulses); %#ok<SAGROW>
        
        
        % Assessing Statistics:
        %         mode_S_quant{ind_fs}   = mode(S_quant);
        
        % Velocity resolution. Frequency resolution: ratio(fs,nfft_points)
        
        f_resolution(ind_fs,ind_npulses) = f{ind_fs}(2); %#ok<SAGROW>
        V_resolution(ind_fs,ind_npulses) = 0.5*lidar_wavelength*(f_resolution(ind_fs,ind_npulses)); %#ok<SAGROW>
        
        
        %         % Relative error in the peak detection
        %         Relative_error(ind_fs,:,ind_npulses) =  100*abs((fd - f_peak(ind_fs,ind_npulses)))/fd;
        %         stdv_signal (ind_fs,:,ind_npulses) = level_noise/sqrt(size(P3,1));
        %         % STDV_signal (ind_fs,:) = level_noise/sqrt(size(P3,1));
        %         RMSE(ind_fs,:,ind_npulses) =sqrt(mean((mean_S{ind_fs}-mean_S_quant{ind_fs}).^2));
        %         SNQR(ind_fs,:,ind_npulses) =((6.02*n_bits-1.25));
        
        % Clear variables
        clear P3
        clear S_quant
        clear S_quant_chop
        clear S_fft_quant
    end
    
    % Signal mean
    Spec_mean_pulse(ind_npulses,:)  = mean(sss(2:end,:),1); %#ok<SAGROW>
    %     S_plot(ind_npulses,:)=sss(4,:);
    % ORiginal spectra
    S_original(ind_npulses,:)=sss(1,:); %#ok<SAGROW>
    
    v_MC_pulse(ind_npulses,:)      = mean(v_MC(2:end,ind_npulses),1); %#ok<SAGROW>
    stdv_v_MC_pulse(ind_npulses,:) = std(v_MC(2:end,ind_npulses)); %#ok<SAGROW>
    RE_pulse(ind_npulses,:)        = stdv_v_MC_pulse(ind_npulses,:)/v_MC_pulse(ind_npulses,:); %#ok<SAGROW>
end
mean_S_original=mean(S_original,1);

%% Statistics

% Frequency
% Original Doppler frequency
mean_fpeak_pulse_OR = mean(f_peak(1,:),1);
stdv_freq_pulse_OR  = std(f_peak(1,:));

% Peaks' statistics
mean_fpeak_pulse    = mean(f_peak(2:end,:),1);
stdv_freq_pulse     = mean(std(f_peak(2:end,:)));
RE_pulse2            = (stdv_freq_pulse ./mean_fpeak_pulse)*100;

%%% Uncertainty in LOS due to bias in sampling frequency
% Mc method
v_s_MC          = mean(v_MC_pulse);
stdv_bias_v_MC  = mean(stdv_v_MC_pulse);
RE_MC           = 100*stdv_bias_v_MC/v_s_MC;
% Analytical method
v_An            = 0.5*lidar_wavelength*mean_fpeak_pulse;
stdv_bias_v_An  = 0.5*sqrt((fd^2*stdv_wavelength^2+lidar_wavelength^2.*stdv_freq_pulse.^2));
RE_v            = (stdv_bias_v_An./v_An)*100;

% Uncertainty in LOS due to averaging of spectra
stdv_av_v_An = std(v_An);
stdv_av_v_MC = std(v_MC_pulse);
RE_v_MC      = (stdv_av_v_MC/v_s_MC)*100;

% Total uncertainty sum of varainces:
stdv_v_T = sqrt((stdv_bias_v_An)^2+(stdv_av_v_An)^2);

% Spec_mean    = mean(Spec_mean_pulse,1);
% mean_f_Peak  = mean(mean_fpeak_pulse,1);

disp(['Uncertainty in V due to sampling frequency and wavelength drift (MC) = ' ,num2str(stdv_bias_v_MC),' m/s'])
disp(['Uncertainty in V due to sampling frequency and wavelength drift (An) = ' ,num2str(stdv_bias_v_An),' m/s'])
disp(['Uncertainty in V due to spectra average (MC) = ',num2str(stdv_av_v_An),' m/s'])
disp(['Uncertainty in V due to spectra average (An) = ',num2str(stdv_av_v_MC),' m/s'])

%% PLOTS

%%%%%%%%% Plotting signals + digitised signal + error: %%%%%%%%%%%%%%%%%%%%
% markersize=[{'k-*'},{'b-*'},{'r-*'},{'g-*'},{'c-*'},{'m-*'},{'y-*'}];
% markersize_2=[{'r--x'},{'b--x'},{'r--x'},{'g--x'},{'c--x'},{'m--x'},{'y--x'}];
% markersize_3=[{'k-.'},{'b-.'},{'r-.'},{'g-.'},{'c-.'},{'m-.'},{'y-.'}];
% figure,hold on
% for in_sig=1:length(fs)
%     Legend1{in_sig}=['fs = ',num2str(fs(in_sig),'%.2s') , '; SNQR = ', num2str(SNQR{in_sig},'%.2s'),' dB'];
%     plot(t{in_sig}(1,:)  ,mean_S{in_sig}(1,:),'Linewidth',1.2,'displayname',Legend1{in_sig})
%     plot(t{in_sig},mean_S_quant{in_sig},'Linewidth',1.9,'displayname','Q');
%     plot(t{in_sig},mean_S{in_sig}(1,:)-mean_S_quant{in_sig},'Linewidth',1.9,'displayname',['Quantization error; N_MC° bits =',num2str(n_bits) ]);
%     %     plot(t{i} ,X{i}(1,:)  ,markersize{i})
%     %     plot(t{i},S_quant0{i},markersize_2{i},'Linewidth',1.4);
% end
% xlabel('time [s]', 'fontsize',20)
% ylabel('[-]', 'fontsize',20)
% title(['Signal -',' sigma = ',num2str(level_noise),'; n°bits = ', num2str(n_bits)],'fontsize',25)
% hold off
% legend show
% grid on
% set(gca,'FontSize',20);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% fft signals %%%%%%%%%%%%%%
fig=figure;
hold on
plot(fv,pow2db(S_original),'HandleVisibility','off');
plot(fv,pow2db(mean_S_original),'-k','linewidth',2.7,'displayname','Averaged Doppler Spectra');
Y_text_in_plot=pow2db(max(mean_S_original)); % Takes the height of the last peak. Just a convention, to plot properly
str={['# fft                  =  ', num2str(n_fftpoints)],...
    ['# pulses           =  ', num2str(n_pulses)],...
    ['# MC samples =  ', num2str(N_MC)],...
    ['\sigma_{v,avg} [ms^{-1}]     =  ', num2str(stdv_av_v_An,'%.1s') ],...
    ['\sigma_{v,bias} [ms^{-1}]    =  ', num2str(stdv_bias_v_An,'%.1s') ],...
    ['\sigma_{v} [ms^{-1}]          =  ', num2str(stdv_v_T,'%.1s') ],...
    ['RE_{v} [%]           =  ', num2str(RE_MC,'%.1s') ]};
% text(5e6,Y_text_in_plot-2.9,str, 'fontsize',17);
anot=annotation(fig, 'textbox');
anot.FontSize=19;
anot.String=str;
anot.Position =  [0.135 0.415 0.6 0.5];
anot.FitBoxToText='on';
% title('Frequency spectra', 'fontsize',25)
xlabel('Frequency [Hz]');
ylabel('PSD [dB]');
% legend show
leg=legend;
leg.FontSize=19 ;
legend show
grid on
% l1.MarkerFaceColor = l1.Color;
set(gca,'FontSize',35);
hold off

toc
%%%%% Plot matlab spectrum %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure, hold on
% for in_fft=1:length(fs)
%
%     plot(fr{in_fft},pow2db(pxx{in_fft}))
%
% end
% legend show
% grid on
% xlabel('Frequency (Hz)')
% ylabel('Power Spectrum (dB)')
% title('Default Frequency Resolution')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% ####### LOG SCALE ########

% set(gca, 'YScale', 'log')
% set(gca, 'XScale', 'log')


% % Plotting sensitivity annalysis
% figure,plot(n_fftpoints,Relative_error,'bo')
% title('Relative error vs. N_MC°of samples', 'fontsize',20)
% ylabel('Relative error [%]', 'fontsize',20)
% xlabel('Number of fft points', 'fontsize',20)
% grid on
% figure,plot(n_fftpoints,T,'bo')
% title('Computational time vs. N_MC°of samples', 'fontsize',20)
% ylabel('time[s]', 'fontsize',20)
% xlabel('Number of fft points', 'fontsize',20)
% legend
% grid on
% set(gca,'FontSize',20);% %
% figure,plot(fs,Relative_error,'bo')
% title('Relative error vs. Sampling frequency', 'fontsize',20)
% ylabel('Relative error [%]', 'fontsize',20)
% xlabel('Sampling frequency [Hz]', 'fontsize',20)
