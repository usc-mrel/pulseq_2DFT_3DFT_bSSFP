% demo_pulseq_bssfp.m
% Written by Nam Gyun Lee, Ziwei Zhao
% 

%% Clean slate
close all; clear all; clc;

%% Set source directories
% package_directory = 'D:\lowfield_spiral_mprage';
% pulseq_directory = 'D:\pulseq\pulseq';
% addpath(genpath('/Users/ziwei/Documents/matlab/bSTAR_seq/lowfield_bstar'));
% addpath(genpath('/Users/ziwei/Documents/matlab/bSTAR_seq/github/Dbstar/Dbstar_seq'));
pulseq_directory = '/Users/sarinakapai/Desktop/USC/Research/MRI/Software Tools/pulseq-master';
addpath(genpath(pulseq_directory));

%% Load an input file
input_pulseq_3d_cartesian_bssfp;

%% Make an output directory
mkdir(output_directory);

%% Set system limits
max_grad_derated = max_grad / sqrt(3);
max_slew_derated = max_slew / sqrt(3);

sys_slew_derated = mr.opts('MaxGrad'  , max_grad        , 'GradUnit', 'mT/m' , ...
                      'MaxSlew'       , max_slew_derated, 'SlewUnit', 'T/m/s', ...
                      'rfRingdownTime', 20e-6 , ...
                      'rfDeadTime'    , 100e-6, ...
                      'adcDeadTime'   , 10e-6 , ...
                      'B0'            , B0);

sys_derated = mr.opts('MaxGrad'       , max_grad_derated, 'GradUnit', 'mT/m' , ...
                      'MaxSlew'       , max_slew_derated, 'SlewUnit', 'T/m/s', ...
                      'rfRingdownTime', 20e-6 , ...
                      'rfDeadTime'    , 100e-6, ...
                      'adcDeadTime'   , 10e-6 , ...
                      'B0'            , B0);

sys = mr.opts('MaxGrad'       , max_grad, 'GradUnit', 'mT/m' , ...
              'MaxSlew'       , max_slew, 'SlewUnit', 'T/m/s', ...
              'rfRingdownTime', 20e-6 , ...
              'rfDeadTime'    , 100e-6, ...
              'adcDeadTime'   , 10e-6 , ...
              'B0'            , B0);

%% 3D Cartesian bSSFP pulse sequence
%-------------------------------------------------------------------------------------------------------|
%                                              TR                                                       |
%    |         |<--------------------------------------------------------------->|                      |
%    |         |          TR / 2 (TR_left)               TR / 2 (TR_right)       |                      |
%    |         |<------------------------------>|<------------------------------>|                      |
%    |         |         |            |         |         |            |         |         |            |
%    | |       |rf     | |            |         |         |            | |       |rf     | |            |
%    | |      _|_      | |            |         |         |            | |      _|_      | |            |
% RF | |     / | \     | |            |         |         |            | |     / | \     | |            |
% ___|_|    /  |  \    |_|____________|_________|_________|____________|_|    /  |  \    |_|____________|
%    | |\__/   |   \__/| |            |         |         |            | |\__/   |   \__/| |            |
%    | |       |       | |            |         |         |            | |       |       | |            |
%    | |       |       | |            |         |         |            | |       |       | |            |
%    | |       |       | |            |         |         |            | |       |       | |            |
%    | |_______|_______| |         |  |         |         |  |         | |_______|_______| |            |
% Gz | /       | gz    \ |gz_rephaser |         |         | gz_rephaser| /       |       \ |            |
% ___|/        |        \|         |__|_________|_________|__|         |/        |        \|         |__|
%    |         |         \         /  |         |         |  \         /         |         \         /  |
%    |         |         |\_______/|  |         |         |  |\_______/|         |         |\_______/|  |
%    |         |         |         |  |         |         |  |         |         |         |            |
%    |<----------------->|<------->|  |         |         |  |<------->|                   |            |
%    |  mr.calcDuration  |   delay1   |         |         |   delay2   |                   |            |
%    |       (gz)        |<---------->|         |         |<---------->|                   |            |
%    |                   |     _____  |                   |  _____     |                   |     _____  |
% Gz |                   |    /_____\ |                   | /_____\    |                   |    /_____\ |
% ___|___________________|___/_______\|___________________|/_______\___|___________________|___/_______\|
%    |                   |   \_______/|                   |\_______/   |                   |   \_______/|
%    |                   |    \_____/ |                   | \_____/    |                   |    \_____/ |
%    |                   |     gz_pe  |                   |  gz_pe     |                   |            |
%    |                   |            |                   |            |                   |            |
%    |                   |            |         |         |            |                   |            |
%    |                   |            |         |         |            |                   |            |
%    |                   |            | |_______|_______| |            |                   |            |
% Gx |                   |            | / |     |gx   | \ |            |                   |            |
% ___|___________________|____        |/| |     |     | |\|        ____|___________________|____        |
%    |                   |    \       / | |     |     | | \       /    |                   |    \       /
%    |                   |     \_____/| | |     |     | | |\_____/     |                   |     \_____/|
%    |                   gx_prephaser |                   | gx_prephaser                   |            |
%    |                   |            |                   |            |                   |            |
%    |                   |            |                   |            |                   |            |
%    |                   |     _____  |                   |  _____     |                   |     _____  |
% Gy |                   |    /_____\ |                   | /_____\    |                   |    /_____\ |
% ___|___________________|___/_______\|___________________|/_______\___|___________________|___/_______\|
%    |                   |   \_______/|                   |\_______/   |                   |   \_______/|
%    |                   |    \_____/ |                   | \_____/    |                   |    \_____/ |
%    |                   |     gy_pe  |                   |  gy_pe     |                   |            |
%    |                   |            |                   |            |                   |            |
%    |<----------------->|<---------->|<----------------->|<---------->|                   |            |
%    |      block 1      |   block 2  |      block 3      |   block 4  |                   |            |
%----o-------------------+------------+-------------------|------------+-------------------|----------> t
%--------------------------------------------------------------------------------------------------------
% NOTE: mr.calcDuration(gz) - (rf.delay + mr.calcRfCenter(rf)) = gz.flatTime / 2 + gz.fallTime
%
% Calculate minimum TE
% min_delay1 = max([gy_pe_duration gz_pe_duration mr.calcDuration(gx_prephaser)])
% minTE = gz.flatTime / 2 + gz.fallTime + min_delay1 + mr.calcDuration(gx) / 2
% TR_left (minTE) = mr.calcDuration(gz) - (rf.delay + mr.calcRfCenter(rf)) + delay1 + mr.calcDuration(gx) / 2
% TR_right (minTE + gz.delay) = mr.calcDuration(gx) / 2 + delay2 + (rf.delay + mr.calcRfCenter(rf))
% TR = TR_left + TR_right
%
% If TR > minTE + (minTE + gz.delay):
% x, x + gz.delay
% 2x + gz.delay = TR
% x = (TR - gz.delay) / 2
% delay1: (TR - gz.delay) / 2 = gz.flatTime / 2 + gz.fallTime + delay1 + mr.calcDuration(gx) / 2
% delay2: (TR - gz.delay) / 2 + gz.delay = mr.calcDuration(gx) / 2 + delay2 + gz.delay + gz.riseTime + gz.flatTime / 2

%% Create a alpha-degree slice selection pulse event [Hz] and corresponding gradient events [Hz/m]

% sinc pulse
[rf, gz] = mr.makeSincPulse(flip_angle * pi / 180, 'Duration', rf_length, 'SliceThickness', slice_thickness * slices_per_slab, 'apodization', rf_apodization, 'timeBwProduct', rf_tbw, 'system', sys_derated);
% SLR pulse

% spin echo pulse





%% Create a slice-selection rephaser gradient event [Hz/m]
gz_rephaser = mr.makeTrapezoid('z', 'Area', -gz.area / 2, 'Duration', gz.flatTime / 2 + gz.fallTime, 'system', sys_derated);

%% Create an alpha/2-degree slice selection pulse event [Hz]
rf_half = mr.makeSincPulse(flip_angle / 2 * pi / 180, 'Duration', rf_length, 'SliceThickness', slice_thickness * slices_per_slab, 'apodization', rf_apodization, 'timeBwProduct', rf_tbw, 'system', sys_derated);

%% Calculate the real dwell time [sec]
%--------------------------------------------------------------------------
% real dwell time [sec]
% IDEA p219: dRealDwellTime denotes the dwell time with oversampling
%--------------------------------------------------------------------------
% round-down dwell time to 100 ns (sys.adcRasterTime  = 100 ns)
real_dwell_time = round((1 / bandwidth) / (readout_os_factor * base_resolution) * 1e7) * 1e-7;

%% Calculate the duration of an ADC event
adc_samples = base_resolution * readout_os_factor; % true number of samples
adc_duration = adc_samples * real_dwell_time;

%% Calculate the duration of a trapezoid (TotalTime) [sec]
flat_time = ceil(adc_duration / (2 * sys.gradRasterTime)) * (2 * sys.gradRasterTime); % [sec]

%% Create a readout gradient event ([PE,RO,SL] = [y,x,z] in Pulseq)
% Siemens GCS (PRS) 'RO' => Pulseq 'x'
deltak_read = 1 / (fov_read * readout_os_factor); % [cycle/m]
%gx = mr.makeTrapezoid('x', 'FlatArea', adc_samples * deltak_read, 'FlatTime', flat_time, 'system', sys_slew_derated);
gx = mr.makeTrapezoid('x', 'FlatArea', adc_samples * deltak_read, 'FlatTime', flat_time, 'system', sys);
gx_prephaser = mr.makeTrapezoid('x', 'Area', -gx.area / 2, 'system', sys_derated);

%% Create a phase-encoding view table ([PE,RO,SL] = [y,x,z] in Pulseq)
% Siemens GCS (PRS) 'PE' => Pulseq 'y'
deltak_phase = 1 / (fov_read * fov_phase * 1e-2); % [cycle/m]
phase_areas = (-floor(nr_phase_encoding_steps_1/2):ceil(nr_phase_encoding_steps_1/2)-1).' * deltak_phase;

%% Create a partition-encoding view table ([PE,RO,SL] = [y,x,z] in Pulseq)
% Siemens GCS (PRS) 'SL' => Pulseq 'z'
deltak_partition = 1 / (slice_thickness * slices_per_slab); % [cycle/m]
partition_areas = (-floor(nr_phase_encoding_steps_2/2):ceil(nr_phase_encoding_steps_2/2)-1).' * deltak_partition; % [cycle/m]

%% Flip the sign of gradients along the PE and SL directions [PE,RO,SL]
%--------------------------------------------------------------------------
% The "old/compat” option maps Pulseq logical X, Y, and Z axes to the three 
% axes of Siemens logical coordinate system (PE-RO-SL) and uses Siemens native 
% transformation from the logical gradient waveforms to the physical gradient
% waveforms. This also involves scaling all gradient axes by -1 followed by 
% additional scaling on the readout direction by -1. To counter these scaling 
% operations performed in the Pulseq interpreter, our code prepares a Pulseq 
% file by intentionally scaling gradient waveforms along the PE and SL 
% directions by -1.
%--------------------------------------------------------------------------
phase_areas = flip(phase_areas,1); % PE
partition_areas = flip(partition_areas,1); % SL

%% Create a phase-encoding gradient event ([PE,RO,SL] = [y,x,z] in Pulseq)
gy_phase_encoding = mr.makeTrapezoid('y', 'Area', max(abs(phase_areas)), 'system', sys_derated);
gy_pe_duration = mr.calcDuration(gy_phase_encoding);

%% Create a partition-encoding gradient event ([PE,RO,SL] = [y,x,z] in Pulseq)
gz_partition_encoding_area_max = gz_rephaser.area + max(partition_areas);
gz_partition_encoding_area_min = gz_rephaser.area + min(partition_areas);
gz_partition_encoding_area_abs_max = max(abs([gz_partition_encoding_area_max gz_partition_encoding_area_min]));

gz_partition_encoding = mr.makeTrapezoid('z', 'Area', gz_partition_encoding_area_abs_max, 'system', sys_derated);
gz_pe_duration = mr.calcDuration(gz_partition_encoding);

%% Calculate the duration of phase-encoding gradient events
pe_duration = max(gy_pe_duration, gz_pe_duration);

%% Calculate timing (need to decide on the block structure already)
%--------------------------------------------------------------------------
% min_delay1 = max([gy_pe_duration gz_pe_duration mr.calcDuration(gx_prephaser)])
% minTE = gz.flatTime / 2 + gz.fallTime + min_delay1 + mr.calcDuration(gx) / 2
% If TR > 2 * minTE + gz.delay:
% x, x + gz.delay
% 2x + gz.delay = TR
% x = (TR - gz.delay) / 2
% delay1: (TR - gz.delay) / 2 = gz.flatTime / 2 + gz.fallTime + delay1 + mr.calcDuration(gx) / 2
% delay2: (TR - gz.delay) / 2 + gz.delay = mr.calcDuration(gx) / 2 + delay2 + gz.delay + gz.riseTime + gz.flatTime / 2
%--------------------------------------------------------------------------
min_delay1 = max([gy_pe_duration gz_pe_duration mr.calcDuration(gx_prephaser)]);
minTE = gz.flatTime / 2 + gz.fallTime + min_delay1 + mr.calcDuration(gx) / 2;
minTR = 2 * minTE + gz.delay;
delay1 = round(((TR - gz.delay) / 2 - (gz.flatTime / 2 + gz.fallTime + mr.calcDuration(gx) / 2)) / sys.gradRasterTime) * sys.gradRasterTime;
delay2 = round(((TR - gz.delay) / 2 + gz.delay - (mr.calcDuration(gx) / 2 + gz.delay + gz.riseTime + gz.flatTime / 2)) / sys.gradRasterTime) * sys.gradRasterTime;

TR_left = gz.flatTime / 2 + gz.fallTime + delay1 + mr.calcDuration(gx) / 2;
TR_right = mr.calcDuration(gx) / 2 + delay2 + gz.delay + gz.riseTime + gz.flatTime / 2;

TR_left + TR_right

if (abs(TR - minTR) < eps)
else
    assert(TR >= minTR);
end

%% Create an ADC readout event
% NOT WORKING?? A BUG?? Here, adc_delay must be a multiple of 1 us (=1000 ns) instead of 100 ns. 
shift_adc = round((flat_time - adc_duration) / 2 / (sys.adcRasterTime * 10)) * (sys.adcRasterTime * 10); % [sec]
adc_delay = gx.riseTime + shift_adc;
adc = mr.makeAdc(adc_samples, 'Dwell', real_dwell_time, 'delay', adc_delay, 'system', sys);

%% Calculate a k-space ordering scheme
nr_phase_encodes = nr_phase_encoding_steps_1 * nr_phase_encoding_steps_2;
order_table = zeros(nr_phase_encodes, 2, 'double');

if strcmp(loop_order, 'par-in-lin')
    nr_phase_encoding_steps_outer_loop = nr_phase_encoding_steps_1;
    nr_phase_encoding_steps_inner_loop = nr_phase_encoding_steps_2;
elseif strcmp(loop_order, 'lin-in-par')
    nr_phase_encoding_steps_outer_loop = nr_phase_encoding_steps_2;
    nr_phase_encoding_steps_inner_loop = nr_phase_encoding_steps_1;
end

outer_order         = (1:nr_phase_encoding_steps_outer_loop).';
inner_order_forward = (1:nr_phase_encoding_steps_inner_loop).';
inner_order_reverse = (nr_phase_encoding_steps_inner_loop:-1:1).';

for idx2 = 1:nr_phase_encoding_steps_outer_loop

    if mod(idx2-1,2) == 1
        if strcmp(reordering, 'Smooth')
            inner_order = inner_order_reverse;
        else
            inner_order = inner_order_forward;
        end
    else
        inner_order = inner_order_forward;
    end

    for idx1 = 1:nr_phase_encoding_steps_inner_loop
        idx = idx1 + (idx2 - 1) * nr_phase_encoding_steps_inner_loop;
        order_table(idx,:) = [inner_order(idx1) outer_order(idx2)];
    end

end

%% Create a sequence object
seq = mr.Sequence(sys);
start_time = tic;

%% Define sequence blocks for an alpha/2-TR/2 sequence
tstart = tic; fprintf('Defining blocks for an alpha/2-TR/2 sequence... ');
rf_phase_rad = rf_phase * pi / 180;
count = 1;

%--------------------------------------------------------------------------
% Set the phase of an RF event and an ADC event
%--------------------------------------------------------------------------
rf_half.phaseOffset = rf_phase_rad * mod(count,2);
adc.phaseOffset = rf_phase_rad * mod(count,2);

%--------------------------------------------------------------------------
% Add a new block to the sequence
%--------------------------------------------------------------------------
seq.addBlock(rf_half, gz);
seq.addBlock(mr.scaleGrad(gz,-1), mr.makeDelay(TR / 2 - mr.calcDuration(gz)));
count = count + 1;
fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));

%% Define sequence blocks for dummy pulses
%--------------------------------------------------------------------------
% Get the line and partition indlices
% order_table = [inner_order outer_order]
%--------------------------------------------------------------------------
if strcmp(loop_order, 'par-in-lin')
    par_index = order_table(1,1);
    lin_index = order_table(1,2);
elseif strcmp(loop_order, 'lin-in-par')
    lin_index = order_table(1,1);
    par_index = order_table(1,2);
end

%--------------------------------------------------------------------------
% Create a phase-encoding gradient event in the PE direction (PRS)
% [PE,RO,SL] = [y,x,z] in Pulseq
%--------------------------------------------------------------------------
gy_phase_encoding_dummy = mr.makeTrapezoid('y', 'Area', phase_areas(lin_index), 'Duration', pe_duration, 'system', sys_derated);

%--------------------------------------------------------------------------
% Create a phase-encoding rewinder gradient event in the PE direction (PRS)
% [PE,RO,SL] = [y,x,z] in Pulseq
%--------------------------------------------------------------------------
gy_phase_rewinder_dummy = mr.scaleGrad(gy_phase_encoding_dummy, -1);

%--------------------------------------------------------------------------
% Create a partition-encoding gradient event in the SL direction (PRS)
% [PE,RO,SL] = [y,x,z] in Pulseq
%--------------------------------------------------------------------------
gz_partition_encoding_area = gz_rephaser.area + partition_areas(par_index);
gz_partition_encoding_dummy = mr.makeTrapezoid('z', 'Area', gz_partition_encoding_area, 'Duration', pe_duration, 'system', sys_derated);

%--------------------------------------------------------------------------
% Create a partition-encoding rewinder gradient event in the SL direction (PRS)
% [PE,RO,SL] = [y,x,z] in Pulseq
%--------------------------------------------------------------------------
gz_partition_rewinder_area = gz_rephaser.area - partition_areas(par_index);
gz_partition_rewinder_dummy = mr.makeTrapezoid('z', 'Area', gz_partition_rewinder_area, 'Duration', pe_duration, 'system', sys_derated);

for i = 1:rf_dummies
    tstart = tic; fprintf('Defining blocks for dummy pulses (%3d/%3d)... ', i, rf_dummies);

    %----------------------------------------------------------------------
    % Set the phase of an RF event and an ADC event
    %----------------------------------------------------------------------
    rf.phaseOffset = rf_phase_rad * mod(count,2);
    adc.phaseOffset = rf_phase_rad * mod(count,2);

    %----------------------------------------------------------------------
    % Add a new block to the sequence (Block 1)
    %----------------------------------------------------------------------
    seq.addBlock(rf, gz);

    %----------------------------------------------------------------------
    % Add a new block to the sequence (Block 2)
    %----------------------------------------------------------------------
    seq.addBlock(mr.align('left', mr.makeDelay(delay1), 'right', gx_prephaser, gy_phase_encoding_dummy, gz_partition_encoding_dummy));

    %----------------------------------------------------------------------
    % Add a new block to the sequence (Block 3)
    %----------------------------------------------------------------------
    seq.addBlock(gx);

    %----------------------------------------------------------------------
    % Add a new block to the sequence (Block 4)
    %----------------------------------------------------------------------
    seq.addBlock(mr.align('right', mr.makeDelay(delay2), 'left', gx_prephaser, gy_phase_rewinder_dummy, gz_partition_rewinder_dummy));
    count = count + 1;
    fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));
end

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
%--------------------------------------------------------------------------
% Average (AVG)
%--------------------------------------------------------------------------
for idx2 = 1:nr_averages

    %----------------------------------------------------------------------
    % Set 'AVG' label
    %----------------------------------------------------------------------
    lbl_set_avg = mr.makeLabel('SET', 'AVG', idx2 - 1);

    for idx1 = 1:nr_phase_encodes

        %------------------------------------------------------------------
        % Get the line and partition indlices
        % order_table = [inner_order outer_order]
        %------------------------------------------------------------------
        if strcmp(loop_order, 'par-in-lin')
            par_index = order_table(idx1,1);
            lin_index = order_table(idx1,2);
        elseif strcmp(loop_order, 'lin-in-par')
            lin_index = order_table(idx1,1);
            par_index = order_table(idx1,2);
        end

        tstart = tic; fprintf('(AVG=%d/%d)(LIN=%3d/%3d)(PAR=%3d/%3d): Defining blocks for bSSFP data acquisitions (%3d/%3d)... ', idx2, nr_averages, lin_index, nr_phase_encoding_steps_1, par_index, nr_phase_encoding_steps_2);

        %------------------------------------------------------------------
        % Set 'LIN' label
        %------------------------------------------------------------------
        lbl_set_lin = mr.makeLabel('SET', 'LIN', lin_index - 1);

        %------------------------------------------------------------------
        % Set 'PAR' label
        %------------------------------------------------------------------
        lbl_set_par = mr.makeLabel('SET', 'PAR', par_index - 1);

        %------------------------------------------------------------------
        % Set the phase of an RF event and an ADC event
        %------------------------------------------------------------------
        rf.phaseOffset = rf_phase_rad * mod(count,2);
        adc.phaseOffset = rf_phase_rad * mod(count,2);

        %------------------------------------------------------------------
        % Create a phase-encoding gradient event in the PE direction (PRS)
        % [PE,RO,SL] = [y,x,z] in Pulseq
        %------------------------------------------------------------------
        gy_phase_encoding = mr.makeTrapezoid('y', 'Area', phase_areas(lin_index), 'Duration', pe_duration, 'system', sys_derated);

        %------------------------------------------------------------------
        % Create a phase-encoding rewinder gradient event in the PE direction (PRS)
        % [PE,RO,SL] = [y,x,z] in Pulseq
        %------------------------------------------------------------------
        gy_phase_rewinder = mr.scaleGrad(gy_phase_encoding, -1);

        %------------------------------------------------------------------
        % Create a partition-encoding gradient event in the SL direction (PRS)
        % [PE,RO,SL] = [y,x,z] in Pulseq
        %------------------------------------------------------------------
        gz_partition_encoding_area = gz_rephaser.area + partition_areas(par_index);
        gz_partition_encoding = mr.makeTrapezoid('z', 'Area', gz_partition_encoding_area, 'Duration', pe_duration, 'system', sys_derated);

        %------------------------------------------------------------------
        % Create a partition-encoding rewinder gradient event in the SL direction (PRS)
        % [PE,RO,SL] = [y,x,z] in Pulseq
        %------------------------------------------------------------------
        gz_partition_rewinder_area = gz_rephaser.area - partition_areas(par_index);
        gz_partition_rewinder = mr.makeTrapezoid('z', 'Area', gz_partition_rewinder_area, 'Duration', pe_duration, 'system', sys_derated);

        %------------------------------------------------------------------
        % Add a new block to the sequence (Block 1)
        %------------------------------------------------------------------
        seq.addBlock(rf, gz, lbl_set_lin, lbl_set_par, lbl_set_avg);

        %------------------------------------------------------------------
        % Add a new block to the sequence (Block 2)
        %------------------------------------------------------------------
        seq.addBlock(mr.align('left', mr.makeDelay(delay1), 'right', gx_prephaser, gy_phase_encoding, gz_partition_encoding));

        %------------------------------------------------------------------
        % Add a new block to the sequence (Block 3)
        %------------------------------------------------------------------
        seq.addBlock(gx, adc);

        %------------------------------------------------------------------
        % Add a new block to the sequence (Block 4)
        %------------------------------------------------------------------
        seq.addBlock(mr.align('right', mr.makeDelay(delay2), 'left', gx_prephaser, gy_phase_rewinder, gz_partition_rewinder));
        count = count + 1;
        fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));
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

%% Prepare sequence export
seq.setDefinition('FOV', [fov_read fov_read * fov_phase * 1e-2 slice_thickness * slices_per_slab]); % [m]
seq.setDefinition('Name', '3D Cartesian bSSFP');

%--------------------------------------------------------------------------
% Contrast (common)
%--------------------------------------------------------------------------
seq.setDefinition('TR', TR);                % TR [sec]
seq.setDefinition('TE', TR / 2);            % TE [sec]
seq.setDefinition('FlipAngle', flip_angle); % Flip angle [deg]

%--------------------------------------------------------------------------
% Resolution (common)
%--------------------------------------------------------------------------
seq.setDefinition('FovRead', fov_read);                 % Fov read [m]
seq.setDefinition('FovPhase', fov_phase);               % Fov phase [%]
seq.setDefinition('SliceThickness', slice_thickness);   % Slice thickness [m]
seq.setDefinition('BaseResolution', base_resolution);   % Base resolution
seq.setDefinition('PhaseResolution', phase_resolution); % Phase resolution [%]

%--------------------------------------------------------------------------
% Geometry (common)
%--------------------------------------------------------------------------
seq.setDefinition('SlicesPerSlab', slices_per_slab); % Number of slices (partitions) per slab

%--------------------------------------------------------------------------
% Sequence (Part 1)
%--------------------------------------------------------------------------
seq.setDefinition('Bandwidth', bandwidth); % Bandwidth [Hz/Px]

%--------------------------------------------------------------------------
% Misc
%--------------------------------------------------------------------------
seq.setDefinition('ReadoutOsFactor', readout_os_factor);
seq.setDefinition('PhaseEncodingSteps1', nr_phase_encoding_steps_1);
seq.setDefinition('PhaseEncodingSteps2', nr_phase_encoding_steps_2);

seq_filename = sprintf('bssfp_3d_%s_TR%3.2fms_%3.0fx%3.0fx%2.0fmm_base%d_slice%d_thk%3.2fmm_%s_osf%d.seq', loop_order, TR * 1e3, fov_read * 1e3, fov_read * fov_phase * 1e-2 * 1e3, slice_thickness * slices_per_slab * 1e3, nr_phase_encoding_steps_1, nr_phase_encoding_steps_2, slice_thickness * 1e3, lower(reordering), readout_os_factor);
seq_file = fullfile(output_directory, seq_filename);
seq.write(seq_file); % Write to a pulseq file

%% plot sequence and k-space diagrams
seq.plot('timeRange', [0 50] * TR, 'label', 'PAR');

if 0
% k-space trajectory calculation
[ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing] = seq.calculateKspacePP();

% plot k-spaces
figure; plot3(ktraj(1,:), ktraj(2,:), ktraj(3,:), 'b'); % a 3D k-space plot
axis('equal'); % enforce aspect ratio for the correct trajectory display
hold; plot3(ktraj_adc(1,:), ktraj_adc(2,:), ktraj_adc(3,:), 'r.'); % plot the sampling points
title('full k-space trajectory (k_x x k_y x k_z)');
end

return

%% very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within slewrate limits  
rep = seq.testReport;
fprintf([rep{:}]);
