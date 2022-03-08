function [dcoeff] = filter_fun(eigval, tmp1, proj_coeff, coeff, W3, W4, i)
%% This is a function that define

dcoeff = cell(1,5);
if i ==1
    dcoeff{5} = (W3*(coeff.')).';
    dcoeff{4} = (W4*(coeff.')).';
else
    dcoeff{5} = (W3*([coeff,conj(coeff)].')).';
    dcoeff{4} = (W4*([coeff,conj(coeff)].')).';
end
dcoeff{1} = (tmp1*bsxfun(@times, eigval, proj_coeff)).';
dcoeff{2} = (tmp1*bsxfun(@times, 2*eigval-eigval.^2, proj_coeff)).';
dcoeff{3} = (tmp1*bsxfun(@times, (2*eigval-eigval.^2).^10, proj_coeff)).';
dcoeff{3} = (tmp1*bsxfun(@times, (2*eigval-eigval.^2).^10, proj_coeff)).';
dcoeff{6} = (tmp1*bsxfun(@times, eigval.^3 - 3*(eigval.^2) + 3*eigval, proj_coeff)).';

end
