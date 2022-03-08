function [c_limit, R_limit, cum_pspec, cum_var, noise_var] = support_estimate( images, threshold )
%% Determine sizes of the compact support in both real and Fourier space.
% OUTPUTS:
% c_limit: Size of support in Fourier space
% R_limit: Size of support in real space

L=size( images , 1 );
N=floor(L/2);
P=size(images , 3);
[ x, y ] = meshgrid(-N:N, -N:N);
r = sqrt(x.^2 + y.^2);
r_max=N;


%img=(icfft2(images))*L;
images = reshape(images, L^2, P);
mean_data = mean(images, 2);
%remove mean from the data
img = bsxfun(@minus, images, mean_data);

% Estimate the noise variance from the corner pixels
img_corner=reshape(images, L^2, P);
img_corner=img_corner(r>r_max, :);
noise_var=var(img_corner(:));

images = reshape(images, L, L, P);
imgf = cfft2(images)/L;
clear images;
variance_map = var(img, [], 2);
variance_map = reshape(variance_map, L, L);
%mean 2D variance radial function
radial_var = zeros(N, 1);
for i = 1:N
    radial_var(i) = mean(variance_map(r>=i-1 & r<i));
end;

img_ps=abs(imgf).^2;
pspec = mean(img_ps, 3);

radial_pspec = zeros(N, 1);
%compute the radial power spectrum;
for i = 1:N
    radial_pspec(i) = mean(pspec(r>=i-1 & r<i));
end;

%subtract the noise variance
%figure; plot(radial_pspec);
radial_pspec = radial_pspec-noise_var;
radial_var = radial_var-noise_var;

radial_pspec(radial_pspec<0) = 0;
radial_var(radial_var<0) = 0;

c = linspace(0,0.5,N)';
R = [0:N-1]';
cum_pspec = zeros(N, 1);
cum_var = zeros(N, 1);
for i = 1:N
    cum_pspec(i) = sum(radial_pspec(1:i).*c(1:i));
    cum_var(i) = sum(radial_var(1:i).*R(1:i));
end;
cum_pspec = cum_pspec/cum_pspec(end);
cum_var = cum_var/cum_var(end);

c_limit = c(find(cum_pspec>threshold, 1)-1);
R_limit = R(find(cum_var>threshold, 1) - 1);

