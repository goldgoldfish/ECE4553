% ECE4553_Project.m
% 
% ECE4553 - Pattern Recongnition
% Authors: Ben W. & Chris T.
% Date Created: Oct-30-2018
% 

%% Clearing

clear
clc

%% Import fMRI Image Data


filename = 'mprage_noface.nii';

brain_image_test = load_nii('mprage_noface.nii');

view_nii(brain_image_test);

%v = make_ana(brain_image_test);
%% 