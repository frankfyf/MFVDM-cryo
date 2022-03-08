function [sPCA_data, sPCA_coeff, basis, fn ] =  data_sPCA_v2(images, noise_v_r, adaptive_support)

n = size(images, 3);
L = size(images, 1);
if nargin < 3 || isempty(adaptive_support)
    c = 0.5;
    R = floor(size(images, 1)/2);
end

c = adaptive_support.c;
R = adaptive_support.R;

n_r = ceil(4*c*R);
[ basis, sample_points ] = precomp_fb( n_r, R, c );
num_pool=10;

log_message('Start computing sPCA coefficients')
[ ~, coeff, mean_coeff, sPCA_coeff, U, D ] = jobscript_FFBsPCA_v2(images, R, noise_v_r, basis, sample_points, num_pool);
log_message('Finished computing sPCA coefficients')

%%The following part is to select components with 95% variance
Freqs = cell(size(D));
RadFreqs = cell(size(D));
for i = 1:size(D)
    if ~isempty(D{i})
        Freqs{i} = (i-1)*ones(length(D{i}), 1);
        RadFreqs{i} = [1:length(D{i})]';
    end
end

Freqs = cell2mat(Freqs);
RadFreqs = cell2mat(RadFreqs);
D = cell2mat(D);
k = min(length(D), 400); %keep the top 500 components
[ D, sorted_id ] = sort(D, 'descend');
D = D(1:k);
Freqs = Freqs(sorted_id(1:k));
RadFreqs = RadFreqs(sorted_id(1:k));
% Keep top 600 components
sCoeff = zeros(length(D), n);
for i = 1:length(D)
    sCoeff(i, :) = sPCA_coeff{Freqs(i)+1}(RadFreqs(i), :);
end

sPCA_data.eigval = D;
sPCA_data.U = U;
sPCA_data.Freqs = Freqs;
sPCA_data.RadFreqs = RadFreqs;
sPCA_data.Coeff = sCoeff;
sPCA_data.Mean = mean_coeff;
sPCA_data.FBcoeff = coeff;
sPCA_data.c=c;
sPCA_data.R=R;

[ fn ] = IFT_FB(R, c);
% log_message('Start reconstructing images after sPCA')
% [~, recon_spca] = denoise_images_analytical(U, fn, mean_coeff, sPCA_coeff, L, R, n);
% log_message('Finished reconstructing images after sPCA')

