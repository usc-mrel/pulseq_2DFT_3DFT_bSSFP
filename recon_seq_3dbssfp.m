%% 3D bSSFP Recon
clear all;
clc;


addpath(genpath('/Users/sarinakapai/Desktop/USC/Research/MRI/Software Tools/arrShow-develop'));
addpath(genpath('/Users/sarinakapai/Desktop/USC/Research/MRI/Software Tools/mapVBVD'));



% root_path = '/Volumes/ZZ-drive/bssfp_experiments/';
% vol_num   = '1031_3dbssfp_cart';
% 
% rawdata_path = fullfile(root_path, vol_num, '/raw_data');
% seq_path     = fullfile(root_path, vol_num, '/seq');
% output_path  = fullfile(root_path, vol_num, '/recon_results');
% 
% cd(rawdata_path);

% read in siemens .dat data
rawdata = dir('meas_MID01284_FID06873_bssfp_3d_par_in_lin_TR20_00ms_230x230x230mm_FA60*.dat');

for i = 1 % : length(rawdata)
    
    tmp_raw=rawdata(i).name;
    obj=mapVBVD(tmp_raw,'ignoreSeg');
    raw2=obj{1,2}.image{''};
    Dim=obj{1,2}.image.sqzDims;

    [nread, nc, npheco, nslices] = size(raw2);
    recon_imgc = zeros(nread, npheco, nc, nslices);

    % ifft along slice dimension
    recon_nslices = fftshift(ifft(ifftshift(raw2, 4), [], 4), 4);

    for inslice = 1:nslices
        for icoil = 1:nc
            recon_imgc(:,:,icoil,inslice) = ifft2c(squeeze(recon_nslices(:,icoil,:,inslice)));
        end
    end
    
    % sos combine:
    recon_img_trufi = squeeze(sqrt(sum(abs(recon_imgc).^2, 3)))*1e3;

    for inslice = 1:nslices
        cmaps_trufi(:,:,:,inslice) = ismrm_estimate_csm_walsh(recon_imgc(:,:,:,inslice), 10); % flipped
    end

    % coil combine using cmaps
    recon_img = squeeze(sum(conj(cmaps_trufi).* recon_imgc, 3)) ./ squeeze(sqrt(sum(abs(cmaps_trufi).^2,3)));

    % display 3D complex image results
    as(recon_img);

end
