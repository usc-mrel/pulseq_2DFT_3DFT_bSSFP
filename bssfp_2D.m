

clc
close all
clear all

pulseq_directory = '/Users/sarinakapai/Desktop/USC/Research/MRI/Software Tools/pulseq-master';

addpath(genpath(pulseq_directory));


%% Define MRI scanner specifications
B0 = 0.55; % main field strength [T]
grad_mode = 'fast';

switch grad_mode
    case 'fast'
        max_grad = 24;      % Max gradient strength [mT/m]
        max_slew = 180.18;  % Maximum slew rate [mT/m/ms]
    case 'normal'
        max_grad = 22;      % Max gradient strength [mT/m]
        max_slew = 100;     % Maximum slew rate [mT/m/ms]
    case 'whisper'
        max_grad = 22;      % Max gradient strength [mT/m]
        max_slew = 50;      % Maximum slew rate [mT/m/ms]
end

%% Set system limits
sys = mr.opts('MaxGrad'       , max_grad, 'GradUnit', 'mT/m' , ...
    'MaxSlew'       , max_slew, 'SlewUnit', 'T/m/s', ...
    'rfRingdownTime', 20e-6 , ...
    'rfDeadTime'    , 100e-6, ...
    'adcDeadTime'   , 10e-6 , ...
    'B0'            , B0);

sys_derated = mr.opts('MaxGrad'       , max_grad/sqrt(3), 'GradUnit', 'mT/m' , ...
    'MaxSlew'       , max_slew/sqrt(3), 'SlewUnit', 'T/m/s', ...
    'rfRingdownTime', 20e-6 , ...
    'rfDeadTime'    , 100e-6, ...
    'adcDeadTime'   , 10e-6 , ...
    'B0'            , B0);


%% Create a sequence object
seq = mr.Sequence(sys);

%% Define imaging parameters

%--------------------------------------------------------------------------
% Parameters for a cartesian trajectory
%--------------------------------------------------------------------------
flip_angle      = 60;        % Flip angle [deg] (reference flip angle)
fov_read        = 230e-3;    % field of view [m]
slice_thickness = 4.5e-3;      % slice thickness [m]
fov_z = slice_thickness;
matrix_size = 192;
resolution = fov_read/matrix_size;       % Base resolution
% 
TR = 4.7e-3;
TE = TR/2;

%--------------------------------------------------------------------------
% Parameters for a windowed sinc pulse (excitation)
%--------------------------------------------------------------------------
rf_length      = 800e-6;       % RF length [sec]
rf_apodization = 0.5;       % RF apodization
rf_tbw         = 1.5;          % RF time bandwidth product [sec*Hz]
rf_phase       = 180;        % RF phase [deg]

% rf_ex : RF pulse waveform
% gz_ex : slice select gradient
% gr_reph: refocusing gradient
[rf_ex, gz_ex, gz_reph] = mr.makeSincPulse(flip_angle * pi / 180, 'Duration', rf_length, 'SliceThickness', fov_z, 'apodization', rf_apodization, 'timeBwProduct', rf_tbw, 'system', sys_derated);

rf_signal = rf_ex.signal;
%--------------------------------------------------------------------------
% Create phase encoding gradient events
%--------------------------------------------------------------------------
delta_Ky = 1/fov_read;
pe_views = (-floor(matrix_size/2):ceil(matrix_size/2)-1).' * delta_Ky;
pe_views = -pe_views; % PE % Flip the sign of a gradient along the PE direction
gy_pre = mr.makeTrapezoid('y', 'Area', max(abs(pe_views)), 'system', sys_derated); % figure out exact duration later
gy_duration = mr.calcDuration(gy_pre);
%--------------------------------------------------------------------------
% Create readout prewinder gradient and rewinder
%--------------------------------------------------------------------------
bandwidth = 789;%700; % [Hz/pixel]
readout_os_factor = 1;
delta_Kx = 1/(fov_read*readout_os_factor); % [cycle/m]
adc_samples = matrix_size*readout_os_factor; % [#pixels]
real_dwell_time = round(1/(bandwidth*adc_samples)*1e7)*1e-7; %[sec]
adc_duration = real_dwell_time*adc_samples;
flat_time = ceil(adc_duration/(2*sys.gradRasterTime))*(2*sys.gradRasterTime);

gx = mr.makeTrapezoid('x','FlatArea',adc_samples*delta_Kx,'FlatTime',flat_time,'system',sys);
gx_pre = mr.makeTrapezoid('x','Area',-gx.area/2,'system',sys_derated);
gx_re = mr.makeTrapezoid('x','Area',-gx.area/2,'system',sys_derated);

%--------------------------------------------------------------------------
% Create ADC event
%--------------------------------------------------------------------------
% round down dwell time to 100 ns
shift_adc = round((flat_time - adc_duration) / 2 / (sys.adcRasterTime * 10)) * (sys.adcRasterTime * 10); % [sec]
adc_delay = gx.riseTime + shift_adc;
adc = mr.makeAdc(adc_samples, 'Dwell', real_dwell_time, 'delay', adc_delay, 'system', sys);

%--------------------------------------------------------------------------
% Create phase encoding rewinder gradient
%--------------------------------------------------------------------------
gy_re = mr.scaleGrad(gy_pre,-1);

%--------------------------------------------------------------------------
% Create slice encoding rewinder gradient
%--------------------------------------------------------------------------
gz_re = gz_reph;


%% Calculate Timing
% this is the smallest we can set TE to be
minTE = mr.calcDuration(gz_ex)/2 + max([mr.calcDuration(gz_re),mr.calcDuration(gy_pre), mr.calcDuration(gx_pre)]) + mr.calcDuration(gx)/2;


% TE = minTE;
% TR = 2*TE;

assert(TE>=minTE);
% what if we want to have a larger TE? where do we insert a "delay" in the
% pulse diagram?
% desired TE - minTE --> this dictacts the amount of time to insert into
% the system
delay1 = TE - mr.calcDuration(gz_ex)/2 - mr.calcDuration(gx)/2;
delay2 = delay1;



% create a sequence object
seq = mr.Sequence(sys);
start_time = tic;

%% Define sequence blocks
% all LABELS / counters and flags are automatically initialized to 0 in the beginning, no need to define initial 0's
% so we will just increment LIN after the ADC event (e.g. during the spoiler)
%--------------------------------------------------------------------------
% ISMRMRD header
% uint16_t kspace_encode_step_1;    /**< e.g. phase encoding line number */
% uint16_t kspace_encode_step_2;    /**< e.g. partition encoding number */
% uint16_t average;                 /**< e.g. signal average number */
% uint16_t slice;                   /**< e.g. imaging slice number */
% uint16_t contrast;                /**< e.g. echo number in multi-echo */
% uint16_t phase;                   /**< e.g. cardiac phase number */
% uint16_t repetition;              /**< e.g. dynamic number for dynamic scanning */
% uint16_t set;                     /**< e.g. flow encoding set */
% uint16_t segment;                 /**< e.g. segment number for segmented acquisition */
% uint16_t user[ISMRMRD_USER_INTS]; /**< Free user parameters */
%--------------------------------------------------------------------------
lbl_inc_lin   = mr.makeLabel('INC', 'LIN', 1); % lin == line
lbl_inc_par   = mr.makeLabel('INC', 'PAR', 1); % par == partition
lbl_inc_avg   = mr.makeLabel('INC', 'AVG', 1); % avg == average
lbl_inc_seg   = mr.makeLabel('INC', 'SEG', 1); % seg == segment (# PEs per shot)
lbl_reset_lin = mr.makeLabel('SET', 'LIN', 0);
lbl_reset_par = mr.makeLabel('SET', 'PAR', 0);
lbl_reset_avg = mr.makeLabel('SET', 'AVG', 0);
lbl_reset_seg = mr.makeLabel('SET', 'SEG', 0);

rf_phase_rad = rf_phase * pi / 180;
count = 1;
%--------------------------------------------------------------------------
% Average (AVG)
%--------------------------------------------------------------------------
nr_averages = 1;
for idx_avg = 1:nr_averages

    %     % reach steady state
    %     for j = 1:200
    %         rf_ex.phaseOffset  = rf_phase_rad * mod(count,2);
    %         adc.phaseOffset = rf_phase_rad * mod(count,2);
    %
    %         % each line in the matrix requires a different phase encoding
    %         % prewinder and rewinder gradient
    %
    %         gy_pre = mr.makeTrapezoid('y', 'Area', pe_views(1), 'Duration', gy_duration, 'system', sys); % constantly fix duration so amplitude scales, not duration
    %         gy_re = mr.scaleGrad(gy_pre,-1);
    %
    %         %--------------------------------------------------------------
    %         % Add a new block to the sequence
    %         %--------------------------------------------------------------
    %         seq.addBlock(rf_ex, gz_ex);
    %         seq.addBlock(mr.align('left', mr.makeDelay(delay1), gz_reph, 'right', gx_pre, gy_pre));
    %         seq.addBlock(gx);
    %         seq.addBlock(mr.align('left', mr.makeDelay(delay2), gx_re, gy_re,'right', gz_re));
    %         count = count + 1;
    %     end

    gy_pre = mr.makeTrapezoid('y', 'Area', pe_views(1), 'Duration', gy_duration, 'system', sys_derated);
    gy_re = mr.scaleGrad(gy_pre, -1);
    nr_prep_pulses = 10;
    for idx1 = 1 : nr_prep_pulses  %linear steady-state catalyzation pulses

        tstart = tic; fprintf('(%d/%d): Defining blocks for linear catalyzation pulses (%d)... ', idx1);
        rf_ex.phaseOffset  = rf_phase_rad * mod(count,2); % Set the phase of an RF event and an ADC event
        adc.phaseOffset = rf_phase_rad * mod(count,2);
        rf_ex.signal = rf_signal / nr_prep_pulses * idx1; % Set the flip angle [degree]
        seq.addBlock(rf_ex, gz_ex);
        seq.addBlock(mr.align('left', mr.makeDelay(delay1), gz_reph, 'right', gx_pre, gy_pre));
        seq.addBlock(gx);
        seq.addBlock(mr.align('left', mr.makeDelay(delay2), gx_pre,  gy_re, 'right', gz_reph));
        count = count + 1;
        fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));
    end




    for i = 1:matrix_size


        rf_ex.phaseOffset  = rf_phase_rad * mod(count,2);
        adc.phaseOffset = rf_phase_rad * mod(count,2);

        % each line in the matrix requires a different phase encoding
        % prewinder and rewinder gradient

        gy_pre = mr.makeTrapezoid('y', 'Area', pe_views(i), 'Duration', gy_duration, 'system', sys); % constantly fix duration so amplitude scales, not duration
        gy_re = mr.scaleGrad(gy_pre,-1);

        %--------------------------------------------------------------
        % Add a new block to the sequence
        %--------------------------------------------------------------
        lbl_set_lin = mr.makeLabel('SET', 'LIN', i-1);

        seq.addBlock(rf_ex, gz_ex, lbl_set_lin);
        seq.addBlock(mr.align('left', mr.makeDelay(delay1), gz_reph, 'right', gx_pre, gy_pre));
        seq.addBlock(gx, adc);
        seq.addBlock(mr.align('left', mr.makeDelay(delay2), gx_re, gy_re,'right', gz_re));
        count = count + 1;




        %----------------------------------------------------------------------
        % Update AVG counter
        %----------------------------------------------------------------------
        if i ~= nr_averages
            seq.addBlock(lbl_inc_avg);
        else
            seq.addBlock(lbl_reset_avg);
        end
    end
end


%% check whether the timing of the sequence is correct
[ok, error_report] = seq.checkTiming;

if (ok)
    fprintf('Timing check passed successfully\n');
else
    fprintf('Timing check failed! Error listing follows:\n');
    fprintf([error_report{:}]);
    fprintf('\n');
end


% bSSFP part
seq.plot('timeRange', [0 450*TR], 'label', 'LIN');

if 0
% k-space trajectory calculation
[ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing] = seq.calculateKspacePP();

% plot k-spaces
figure; plot(ktraj(1,:), ktraj(2,:), 'b'); % a 2D k-space plot
axis('equal'); % enforce aspect ratio for the correct trajectory display
hold; plot(ktraj_adc(1,:), ktraj_adc(2,:), 'r.'); % plot the sampling points
title('full k-space trajectory (k_x x k_y)');
end

%% very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within slewrate limits  
rep = seq.testReport;
fprintf([rep{:}]);


%% Prepare sequence export
seq.setDefinition('BaseResolution', matrix_size);
seq.setDefinition('FOV', [fov_read fov_read slice_thickness]);
seq.setDefinition('Name', 'Cartesian 2D bSSFP');
seq.setDefinition('Resolution', resolution);

seq_filename = sprintf('./bssfp_res%.1fmm_te%.1fms_tr%.1fms.seq', resolution*1e3, TE*1e3, TR*1e3);
output_directory =  '/Users/sarinakapai/Desktop/USC/Research/MRI/Starter Project/Pulseq 3DFT/bSSFP_2D';
seq_path = fullfile(output_directory, seq_filename);
seq.write(seq_path); % Write to a pulseq file
