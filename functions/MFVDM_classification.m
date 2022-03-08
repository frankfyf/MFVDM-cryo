%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MFVDM classification for cryo-EM images
% 
% Inputs:
%   knn_in: number of nearest neighbors for building the graph
%   knn_out: number of output nearest neighbors
%   class: the initial nearest neighbor search result, which is an N by K 
%       matrix, N is the number of images, K is the number of nearest 
%       neighbors (assumed K >= knn_in) class(i,j) is the j-th nearnest
%       neighbor of the i-th node
%   rot: the initial rotational alignment result, the same size as class,
%       rot(i,j) is the alignment between the i-th node and its j-th
%       neighbor
%   refl: indicator if each pair of nearest neihgbors comes from a
%   	reflection. refl(i,j) = 1 indicates the j-th neighbor of the i-th
%   	node is not a reflection, otherwise refl(i,j) = 2.
%   eigen_num: a vector of length k_max (the maximum frequency), the k-th
%       entry represents the truncation at frequency k
% 
% Outputs:
%   sort_nodes: an n by knn_out matrix which indicates MFVDM nearest
%       neighbor search
%   sort_refl: an n by knn_out matrix which is similar as refl to class 

function [sort_nodes, sort_refl, Evec, Eval] = MFVDM_classification(knn_in, knn_out, class, rot, refl, eigen_num)

n = size(class,1); % number of images

class = class(:,1:knn_in);
list = [repmat([1:n]', knn_in, 1), class(:)];
refl = refl(:,1:knn_in);
refl = refl(:);
rot = rot(:,1:knn_in);
rot = rot(:);

%%% Construct the list includes reflection
% divede into two cases(reflection and non-reflection)
list_nrefl = list(refl == 1,:);
list_refl = list(refl == 2,:);
angle_nrefl = rot(refl == 1,:);
angle_refl = rot(refl == 2,:);

% non-reflection case
[ list_nrefl, id ] = sort(list_nrefl,2,'ascend'); % let (m,n) with m<=n 
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
    Evec{i} = sorted_eigvec;
    Eval{i} = sorted_eigval;
end

%%% Calculate the affinities 
V = cell(2*n,k_max);
for i = 1:2*n
    for j = 1:k_max
        Evec_k = Evec{j}(i,1:eigen_num(j))';
        Eval_k = Eval{j}(1:eigen_num(j),1);
        V{i,j} = ((Eval_k*Eval_k').^2).*(Evec_k*Evec_k');
    end
end

len_v = eigen_num.^2;
affinity = zeros(sum(len_v),2*n);
for i = 1:2*n
    num = 0;
    for j = 1:k_max
        affinity(num+1:num+len_v(j),i) = reshape(V{i,j}(1:eigen_num(j),1:eigen_num(j)),len_v(j),1);
        num = num+len_v(j);
    end
    affinity(:,i) = affinity(:,i)/norm(affinity(:,i),2);
end
affinity = affinity'*affinity;
[~, sort_nodes] = sort(affinity(1:n, :), 2, 'descend');

sort_nodes = sort_nodes(:, 2:knn_out+1);
sort_refl = (sort_nodes>n) + 1;
sort_nodes = mod(sort_nodes-1, n) + 1;

end

