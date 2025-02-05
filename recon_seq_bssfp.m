%% bSSFP 3D Cartesian Recon



clear all;
clc;

% add dependencies
addpath(genpath('/Users/sarinakapai/Desktop/USC/Research/MRI/Software Tools/mapVBVD'));

% read in siemens .dat data
rawdata = dir('meas_MID00339_FID09994_pulseq_empty.dat');
filename=rawdata.name;
obj=mapVBVD(filename,'ignoreSeg');
raw2=obj{1,2}.image{''};                    % Kx x Ncoil x Ky x Kz
Dim=obj{1,2}.image.sqzDims;

[nread, nc, npheco, npart] = size(raw2);
recon_imgc = zeros(nread, npheco, nc, npart);       % Kx x Ky x Ncoil x Kz

% take fft along kz direction
recon_imgc = fftshift(ifft(ifftshift(squeeze(raw2)),[],4));

% now take 2D fft along kx ky direction
recon_imgc_3d = zeros(nread, npheco, npart, nc);
for icoil = 1:nc
    for ipart = 1:npart
        temp = ifft2c(squeeze(recon_imgc(:,icoil,:,ipart)));
        recon_imgc_3d(:,:,icoil,ipart) = temp;
    end

end

cmaps = zeros(size(recon_imgc_3d));
for ipart = 1:npart
    % kx x ky x ncoils
    cmaps(:,:,ipart,:) = ismrm_estimate_csm_walsh(squeeze(recon_imgc_3d(:,:,ipart,:)), 32);

end

%cmaps = ismrm_estimate_csm_mckenzie(recon_imgc);

% coil combine - SOS 
recon_img = squeeze(sqrt(sum(abs(recon_imgc_3d).^2, 4)));

% coil combine using cmaps
%recon_imgc  = recon_imgc(:,:,:,2);
recon_img = squeeze(sum(conj(cmaps).* recon_imgc_3d, 4)) ./ squeeze(sqrt(sum(abs(cmaps).^2,4)));




% things to check
% 1. check that peak of kx, ky, kz block has peak (dc) term
% 2. check for line dc in kx, ky, z block after ifft single dim
% 3. check simple kx ky plane to see if it looks like kspace data
% 4. maybe try 3d ifft for kx ky kz block
% sum along coil dimensions for quick check







mag = figure;
set(mag , 'Position', [100 777 100 100]);
title('Magnitude');
temp = recon_img(:,:,32);
imshow(imrotate(abs(temp), 90), [0 max(abs(temp(:)))*0.8]);
saveas(mag, [output_path, '/', filename(1:end-4), '_mag.png']);

pha = figure;
set(pha , 'Position', [100 777 100 100]);
title('Phase');
imshow(imrotate(angle(temp), 90), []); colorbar; colormap(turbo);

saveas(pha, [output_path, '/', filename(1:end-4), '_phase.png']);

