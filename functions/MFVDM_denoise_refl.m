%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MFVDM denoising for cryo-EM images
% 
% Inputs:
%   knn: number of nearest neighbors for denoising
%   class: MFVDM nearest neighbor serach result
%   rot: MFVDM rotational alignment result
%   refl: the reflection indicator of class
%   coeff: the Fourier-Bessel coefficient
%   mean_coeff: the mean of the Fourier-Bessel coefficient
%   fn: IFT of FB basis
%   L: size of the denoise image (L by L)
%   eigen_num: a vector of length k_max (the maximum frequency), the k-th
%       entry represents the truncation at frequency k
% 
% Outputs:
%   image_denoise: denoised image, currently it is a 5 by 1 cell where each 
%       cell contains denoised image result from a specific filter. 

function [ image_denoise ] = MFVDM_denoise_refl(knn, class, rot, refl, coeff, mean_coeff, fn, L, eig_trun)
n = size(coeff{1}, 2);

class = class(:,1:knn);
list = [ repmat([1:n]', knn, 1), class(:)];
refl = refl(:,1:knn);
refl = refl(:);
rot = rot(:,1:knn);
rot = rot(:);
max_order = size(coeff, 1) - 1;

%%% Construct the list includes reflection
% divede into two cases(reflection and non-reflection)
list_nrefl = list(refl == 1,:);
list_refl = list(refl == 2,:);
angle_nrefl = rot(refl == 1,:);
angle_refl = rot(refl == 2,:);

% non-reflection case
[ list_nrefl, id ] = sort(list_nrefl,2,'ascend'); % let (m,n) m<=n 
angle_nrefl(id(:,1) == 2) = -angle_nrefl(id(:,1) == 2); 
[ list_nrefl, id2 ] = unique(list_nrefl,'rows'); % illiminate repeat neighbors
angle_nrefl = angle_nrefl(id2);
list_nrefl = [ list_nrefl; list_nrefl+n ]; 
angle_nrefl = [-angle_nrefl; angle_nrefl ];

% reflection case
[ list_refl, ~ ] = sort(list_refl,2,'ascend'); % let(m,n) m<=n
[ list_refl, id2 ] = unique(list_refl,'rows'); % illiminate repeat neighbors
angle_refl = angle_refl(id2);
list_refl = [ list_refl(:,1), list_refl(:,2)+n; list_refl(:,2), list_refl(:,1)+n ];
angle_refl = [180-angle_refl;180-angle_refl ];

% then case when k = 0
[ list, ~ ] = sort(list,2,'ascend');
[ list, ~ ] = unique(list,'rows');

%%% Construct the MFVDM matrices for denoising
% construct list and angles
I_0 = list(:,1);
J_0 = list(:,2);
I_1tok = [list_nrefl(:,1); list_refl(:,1)];
J_1tok = [list_nrefl(:,2); list_refl(:,2)];
angle_1tok = [ angle_nrefl; angle_refl ];

coeff{1} = bsxfun(@minus,coeff{1}, mean_coeff);
dcoeff = cell(max_order+1,5);
parfor i = 1 : max_order+1
    %sparse matrix
    if(i == 1)
        W = sparse(I_0, J_0, 1, n, n);
    else
        W = sparse(I_1tok, J_1tok, exp(sqrt(-1)*(i-1)*angle_1tok*pi/180), 2*n, 2*n);
    end
    W = W+W';
    H = abs(W);
    D = sum(H, 2);
    % compute the similar matrix
    W3 = bsxfun(@times, 1./D,W);
    W4 = 2*W3 - W3*W3;
    W2 = bsxfun(@times, 1./sqrt(D), W);
    W2 = bsxfun(@times, 1./sqrt(D).', W2);
    % eigen-decomposition 
    [ eigvec, eigval ] = eigs(W2, eig_trun);
    [ eigval, id  ] = sort(real(diag(eigval)), 'descend');
    if min(eigval)<0
        id = find(eigval>abs(min(eigval)));
        eigval = eigval(id);
    end
    eigvec = eigvec(:, id);
    tmp1 = bsxfun(@times, 1./sqrt(D), eigvec);
    tmp2_1 = bsxfun(@times, sqrt(D), eigvec);
    % if k!=0 then extend the coefficients to 2n
    if(i == 1)    
        proj_coeff = tmp2_1'*coeff{i}.';
    else
        proj_coeff = tmp2_1'*[coeff{i},conj(coeff{i})].';
    end
    dcoeff_tmp = filter_fun(eigval, tmp1, proj_coeff, coeff{i}, W3, W4, i);
    % Use different denoising filters
    for j = 1:5
        dcoeff(i,j) = dcoeff_tmp(j);
        if(i~=1)
            dcoeff{i,j} = dcoeff{i,j}(:,1:n); % truncate the coefficients to n dimentions
        end
    end
end

%%% Reconstruction from denoised Fourier-Bessel coeff
image_denoise = cell(1,5);
l = size(fn{1}, 1);
N = size(coeff{1}, 2);
origin = floor(L/2) + 1;
tmp1 = fn{1};
tmp1 = reshape(tmp1, l^2, size(tmp1, 3));
R = size(fn{1}, 1)/2;
for j = 1:5
    tmp2 = tmp1*bsxfun(@plus, mean_coeff, dcoeff{1,j});
    I = reshape(real(tmp2), l, l, N);
    for k = 1:max_order
	tmp = fn{k+1};
	tmp = reshape(tmp, l^2, size(tmp, 3));
	tmp2_pos = tmp*dcoeff{k+1,j};
	tmp2_pos = 2*real(tmp2_pos);
	I = I + reshape(tmp2_pos, l, l, N);
    end
    image_denoise{1,j} = zeros( L, L, N );
    image_denoise{1,j}(origin-R:origin+R-1, origin-R:origin+R-1, :) = real(I);
end

end
