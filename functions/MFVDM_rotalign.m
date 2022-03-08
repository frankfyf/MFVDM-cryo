%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MFVDM rotational alignment for cryo-EM images 
% 
% Inputs:
%   knn_in: number of nearest neighbors for building the graph
%   knn_out: number of output nearest neighbors
%   MFVDM_class: the MFVDM nearest neighbor search (see the function
%       MFVDM_classificaion)
%   MFVDM_refl: the reflection indicator of MFVDM_class
%   init_class: the initial nearest neighbor search  
%   init_refl: the reflection indicator of init_class
%   init_rot: the initial rotational alignment
%   eigen_num: a vector of length k_max (the maximum frequency), the k-th
%       entry represents the truncation at frequency k
% 
% Outputs:
%   opt_align: the MFVDM rotational alignment (in degrees)

function [opt_align] = MFVDM_rotalign(knn_in, knn_out, MFVDM_class, MFVDM_refl, init_class, init_refl, init_rot, eigen_num)

n = size(init_class, 1);

%%% Initialization
sPCA_c = init_class(:,1:knn_in);
list = [repmat([1:n]', knn_in, 1), sPCA_c(:)];
refl = init_refl(:,1:knn_in);
refl = refl(:);
angle = init_rot(:,1:knn_in);
angle = angle(:);
MFVDM_class = MFVDM_class+n*(MFVDM_refl == 2);

%%% Construct the list includes reflection
% divede into two cases(reflection and non-reflection)
list_nrefl = list(refl == 1,:);
list_refl = list(refl == 2,:);
angle_nrefl = angle(refl == 1,:);
angle_refl = angle(refl == 2,:);

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


%%% Construct the affinity matrices with different frequencies
% construct list and angles
I_1tok = [list_nrefl(:,1); list_refl(:,1)];
J_1tok = [list_nrefl(:,2); list_refl(:,2)];
angle_1tok = [ angle_nrefl; angle_refl ];

k_max = numel(eigen_num);
exp_alpha = cell(k_max);
parfor i = 1: k_max
    W = sparse(I_1tok, J_1tok, exp(sqrt(-1)*(i)*angle_1tok*pi/180), 2*n, 2*n); % sparse matrix
    W = W + W';
    H = abs(W);
    D = sum(H, 2);
    W = bsxfun(@times, 1./sqrt(D), W);
    W = bsxfun(@times, 1./sqrt(D).', W);
    [ u, d ] = eigs(W, max(50,eigen_num(i)));
    [ sorted_eigval, id ] = sort(real(diag(d)), 'descend');
    sorted_eigvec = u(:, id);
    Evec = sorted_eigvec;
    Eval = sorted_eigval;
    for i1 = 1:n
        for j1 = 1:knn_out
            exp_alpha{i}(i1,j1) = ((Eval'.^4).*Evec(i1,:))*(Evec(MFVDM_class(i1,j1),:)');
        end
    end        
end

%%% Rotational alignment estimated by FFT
 
N = 1024; % FFT points
opt_align = zeros(n,knn_out);
if(k_max ==1)
    for i = 1:n
        for j = 1:knn_out
            exp_alpha{1}(i,j) = exp_alpha{1}(i,j)/abs(exp_alpha{1}(i,j));
            opt_align(i,j) = 360*real((-1i)*log(exp_alpha{1}(i,j)))/(2*pi);
        end
    end
else
for i = 1:n
    for j = 1:knn_out
	v_1 = zeros(1,k_max);
	for k = 1:k_max
	    v_1(k) = exp_alpha{k}(i,j);
	end
	freq = fft(v_1,N);
	opt_align(i,j) = 360*(find(freq == max(freq))-1)/N;
    end
end
end
opt_align = opt_align - 360*(opt_align > 180);

opt_align = -opt_align;
opt_align(MFVDM_refl == 2) = opt_align(MFVDM_refl == 2) - 180;

end

