% ECE4553_Project.m
% 
% ECE4553 - Pattern Recongnition
% Authors: Ben W. & Chris T.
% Date Created: Oct-30-2018
% 
% All data is from: https://www.nitrc.org/frs/?group_id=383

%% Clearing

clear
clc

%% Import fMRI Image Data

filename = 'mprage_noface.nii';

brain_image_test = load_nii('mprage_noface.nii');

view_nii(brain_image_test);

%v = make_ana(brain_image_test);
%% Loading 1D Data

%{
For this operation the data sets should be in a folder called OneD_datain
a folder that is in the Path. Additionally the files should have a .txt
extension.

The phenotype information is contained in an xlsx file created from the tsv
the adhd200_preprocessed_phenotypics.tsv pheonotype file.
%}

clear name

OneD_names = dir('OneD_data/*.txt');
fMRI_data = cell(size(OneD_names,1),1);

fMRI_info = importdata('adhd200_preprocessed_phenotypics.xlsx');

for i=1:size(OneD_names)
    str_load = OneD_names(i).name;
    temp = importdata(str_load);
    fMRI_data{i,2} = temp.data;
    temp_name = OneD_names(i).name;
    n = 1;
    m = 1;
    writing_name = 2;
    seen_nonzero = 0;
    while (n <= size(temp_name,2))
        if ((temp_name(n) >= '0') && (temp_name(n) <= '9'))
            if (temp_name(n) >= '1' && (temp_name(n) <= '9'))
                name(i,m) = temp_name(n);
                m = m + 1;
                seen_nonzero = 1;
                writing_name = 1;
            elseif (temp_name(n) == '0' && seen_nonzero)
                name(i,m) = temp_name(n);
                m = m + 1;
                writing_name = 1;
            end 
        end
        if (writing_name == 0)
            break;
        end
        if (writing_name == 2)
        else
            writing_name = 0;
        end
        n = n + 1;
    end
    fMRI_data{i,3} = str2num(name(i,:));
    %i
end

%% Matching Datasets with a Diagnosis
x = fMRI_info.data;

for j=1:size(fMRI_data,1)
    index = find(x(:,1) == fMRI_data{j,3});
    fMRI_data{j,1} = x(index,6);
end

%% Clean Variables
clear i
clear index
clear j 
clear m 
clear n
clear name
clear OneD_names
clear str_load
clear writing_name
clear name
clear y
clear x
clear temp
clear temp_name
clear seen_nonzero
clear str_load


%% Finding Good Features

x = 1:size(fMRI_data{i,2},1);
x =x';
temp_plot = fMRI_data{1,2};
temp_plot2 = fMRI_data{2,2};

for i=1:size(fMRI_data{1,2},2)
    figure(1)
    plot(x, temp_plot(:,i), '-o');
    figure(2)
    plot(x, temp_plot2(:,i), '-o');
end
