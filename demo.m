%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is an example of the numerical experiment in the paper "Cryo-Electron 
% Microscopy Image Denoising Using Multi-Frequency Vector Diffusion Maps." 
% by Yifeng Fan and Zhizhen Zhao.
%
% Yifeng Fan (yifengf2@illinois.edu), Feb 2022

addpath(genpath('/home/yifengf2/aspire')); % Add the aspire package to the path
addpath(genpath('./functions')); 
rng('default');

%% Load dataset
file_name = '/home/yifengf2/projects/MFVDM_cryo_EM_denoise/test_images.mat'; % Replace this with your dataset
load(file_name);
raw_images = raw_images(:,:,1:10000); % Assuming the dataset is named 'raw_images'

d = size(raw_images,1); 
n = size(raw_images,3);
    
%% Preprocessing
log_message('Start Preprocessing')

% Estimate the support of the object and the noise variance
threshold = 0.99; % The threshold, usually 0.99 or 0.95
[ c, R, ~, ~, noise_var] = support_estimate(raw_images, threshold);

support.c = c;
support.R = R;
log_message('Finish Preprocessing')

%% Initial classification (nearest neighbor search) and rotational alignment by ASPIRE package
if isempty(gcp('nocreate'))
    parpool('local',12) 
end
    
log_message('Start FBsPCA')
knn = 50; % number of nearest neighbors of each node
[sPCA_data, ~, ~, fn] =  data_sPCA_v2(raw_images, noise_var, support);
[class.sPCA, refl.sPCA, rot.sPCA, ~, ~] = Initial_classification_FD_v2(sPCA_data, knn);
log_message('Finish FBsPCA')


%% MFVDM Classification (nearest neighbor search) and rotational alignment

% Parameter setting
knn_in = 50; % number of nearest neighbors of each node from the initial classification
knn_out = 50; % number of output nearest neighbors
k_max = 10; % maximum frequency
m_trun = 50; % truncation of each frequency
eigen_num = m_trun*ones(1, k_max); % number of eigenvalues that used in rotation alignment

% Classificaiton
log_message('Start MFVDM classification')
[class.MFVDM, refl.MFVDM, ~, ~] = MFVDM_classification(knn_in, knn_out, class.sPCA, rot.sPCA, refl.sPCA, eigen_num);
log_message('Finish MFVDM classification')

% Rotational alignment
log_message('Start MFVDM rotation alignment')
rot.MFVDM = MFVDM_rotalign(knn_in, knn_out, class.MFVDM, refl.MFVDM, class.sPCA, refl.sPCA, rot.sPCA, eigen_num);
log_message('Finish MFVDM rotation alignment')

%% MFVDM denoising

% Parameter setting
knn = 50; % number of nearest neighbors for denoising
m_trun = 50; 

log_message('Start MFVDM denoising')
image_denoise = MFVDM_denoise_refl(knn, class.MFVDM, rot.MFVDM, refl.MFVDM, sPCA_data.FBcoeff, sPCA_data.Mean, fn, d, m_trun);
log_message('Finish MFVDM denoising')


% Visualize the result
for i = 1:5
    plotfaces(reshape(image_denoise{i}(:,:,1:50),l^2,50), 5, 10, 1, l);
end









