%% create a filter with the note which center is in (681,208) in 1.jpeg (before resize) 
clear all;
close all;
clc;

im = rgb2gray(im2double(imread('1.jpeg')));
imR = 4;
im = imresize(im,imR);
figure;
imshow(im);

N = 50*imR;
N_half = floor(N/2);

%% half note and full note
cut_mid =[681*imR, 208*imR]; %[510*imR, 369*imR]; %[1098*imR, 194*imR]; 


cut_left = cut_mid(1) - N_half;
cut_right = cut_mid(1) + N_half - 1;
cut_up = cut_mid(2) - N_half;
cut_down = cut_mid(2) + N_half - 1;
half_note = im(cut_up:cut_down, cut_left:cut_right);

mask_half = half_note > 0.5;

mask_full = imfill(mask_half, 'holes');
figure;
imshow(mask_half);
figure;
imshow(mask_full);
%% define binary filter
mask_full_dilate = imdilate(mask_full, strel('disk',10*imR));
figure;
imshow(mask_full_dilate);

mask_pos = mask_half;
mask_in = mask_full & ~mask_half;
mask_out = ~mask_full_dilate;
mask_neg = mask_full_dilate & ~mask_full;
mask_non_zero = mask_pos|mask_neg;

%% make filter smooth
dist = bwdist(~mask_pos)-0.5;
dist = dist/max(dist(:));
mask_pos_smooth = zeros(200,200);
mask_pos_smooth(mask_pos) = dist(mask_pos);
mask_pos_smooth(mask_pos) = mask_pos_smooth(mask_pos).^0.2;

figure;
imshow(mask_pos_smooth,[]);
dist = bwdist(~mask_neg)-0.5;
dist = dist/max(dist(:));
mask_neg_smooth(mask_neg) = dist(mask_neg);
mask_neg_smooth(mask_neg) = -(mask_neg_smooth(mask_neg).^0.2);

%% normalize filter
s = -sum(mask_neg_smooth(mask_neg(:)))/sum(mask_pos_smooth(mask_pos(:)));

mask_pos_smooth = mask_pos_smooth*s;
H = zeros(N);
H(mask_pos) = mask_pos_smooth(mask_pos);
H(mask_neg) = mask_neg_smooth(mask_neg);

figure;
imshow(H,[]);
full_note_filter = H;
save('full_note_filter.mat','full_note_filter');
sum(H(:))
stop
%% run on im 1

im_filt = filter2(H, im);
figure;
imshow(im_filt,[]);

BW = imregionalmax(im_filt);
BW_2 = im_filt > max(im_filt(:))/2;
BW_3 = filter2(mask_pos|mask_in,BW&BW_2);
figure;
imshow(BW_2)
figure;
imshow(BW_3,[]);

%% show results

im_r = im;
im_g = im;
im_b = im;

im_r(BW_3 > 0) = 0;
im_g(BW_3 > 0) = 0;
im_res = cat(3, im_r, im_g, im_b);
figure;
imshow(im_res);
%% scale im2
im2 = rgb2gray(im2double(imread('3.jpeg')));
imR = 20;
im2 = imresize(im2, imR);
% figure;
% imshow(im2);
% cut_mid = [512*imR, 95*imR];
% 
% N_half_2 = N_half;
% cut_left = cut_mid(1) - N_half_2;
% cut_right = cut_mid(1) + N_half_2 - 1;
% cut_up = cut_mid(2) - N_half_2;
% cut_down = cut_mid(2) + N_half_2 - 1;
% full_note_2 = im2(cut_up:cut_down, cut_left:cut_right);
% 
% figure;
% imshow(full_note_2);

%% run on im 2
im_filt = filter2(H, im2);
figure;
imshow(im_filt,[]);

BW = imregionalmax(im_filt);
BW_2 = im_filt > sum(mask_pos(:))/3;
BW_3 = filter2(mask_pos|mask_in,BW&BW_2);
figure;
imshow(BW_3,[]);

%% show results

im_r = im2;
im_g = im2;
im_b = im2;

im_r(BW_3 > 0) = 0;
im_g(BW_3 > 0) = 0;
im_res = cat(3, im_r, im_g, im_b);
figure;
imshow(im_res);