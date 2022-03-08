function [class, class_refl, rot, corr, timing] = Initial_classification_FD_v2(sPCA_data, n_nbor)
%Description:
%This function does initial classfication on preprocessed data.
%   Input: 
%           sPCA: steerable PCA data.
%           n_nbor: number of nearest neighbors used for class averaging.
%           isrann: using randomized algorithm for finding k nearest
%           neighbors. isrann=0: brute force nearest neighbor search.
%           isrann=1: randomized nearest neighbor search
%   Output:
%           class: a matrix of size n x n_nbor. This provides a list of
%           nearest neighbors for each image.
%           rot: rotational alginment
%           corr: normalized cross correlation between nearest neighbors
%           timing: timing of different steps in the algorithm
%	    FBsPCA_data: FBsPCA denoising data. It includes r_max--radius of 
%           region of interest, UU--eigenvectors, Freqs--associated angular 
%           frequence, Coeff--image expansion coefficients on UU, Mean--estimated
%           rotationally invariant mean of the data, and W--weight for wiener type filtering.
% Zhizhen Zhao Updated Jan 2015


%Coeff = [sPCA_data.Coeff, conj(sPCA_data.Coeff)]; % Tejal April 2016
Coeff = sPCA_data.Coeff;
Freqs = sPCA_data.Freqs;
eigval = sPCA_data.eigval;
clear sPCA_data;

n_im = size(Coeff, 2);

%normalize the coefficients
Coeff(Freqs==0, :) = Coeff(Freqs==0, :)/sqrt(2);
for i=1:n_im  
    Coeff(:, i) = Coeff(:, i)/norm(Coeff(:, i));
end
Coeff(Freqs==0, :) = Coeff(Freqs==0, :)*sqrt(2);

%Compute bispectrum
[ Coeff_b, Coeff_b_r, toc_bispec ] = Bispec_2Drot_v2(Coeff, Freqs, eigval);

% Nearest neighbor search (Brute force)
tic_nn=tic;
corr=real((Coeff_b(:, 1:n_im))'*[Coeff_b, Coeff_b_r]); % Tejal April 21 2016 %Change back from Tejal's modification
corr = corr - sparse(1:n_im, 1:n_im, ones(n_im, 1), n_im, 2*n_im);
[~, class] = sort(corr(1:n_im, :), 2, 'descend');
class = class(:, 1:n_nbor);
toc_nn = toc(tic_nn);
clear Coeff_b Coeff_b_r

% Rotational alignment for nearest neighbor pairs
k_max=max(Freqs);
Cell_Coeff=cell(k_max+1, 1);
for i=1:k_max+1
    Cell_Coeff{i}=[Coeff(Freqs==i-1, :), conj(Coeff(Freqs==i-1, :))]; %Generate the reflected images
end
list = [class(:), repmat([1:n_im]', n_nbor, 1)];

% Initial in-plane rotational alignment within nearest neighbors
tic_rot=tic;
[ corr, rot ] = rot_align_v2(max(Freqs), Cell_Coeff, list );
toc_rot = toc( tic_rot );
corr = reshape(corr, n_im, n_nbor);
rot = reshape(rot, n_im, n_nbor);
class = reshape(class, n_im, n_nbor);
[corr, id_corr] = sort(corr, 2, 'descend');

for i=1:n_im
    class(i, :) = class(i, id_corr(i, :));
    rot(i, :) = rot(i, id_corr(i, :));
end

class_refl=ceil(class/n_im);
class(class>n_im)=class(class>n_im)-n_im;

rot(class_refl==2) = mod(rot(class_refl==2)+180, 360);

timing.bispec=toc_bispec;
timing.nn=toc_nn;
timing.rot=toc_rot;

