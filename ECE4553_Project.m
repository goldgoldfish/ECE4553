% ECE4553_Project.m
% 
% ECE4553 - Pattern Recongnition
% Authors: Ben W. & Chris T.
% Date Created: Oct-30-2018
% 
% All data obtained from: https://www.nitrc.org/frs/?group_id=383

%% Clearing

clear
clc

%% Import fMRI Image Data

%filename = 'ADHD200_parcellate_400.nii';

brain_image_test = load_nii('ADHD200_parcellate_400.nii');

view_nii(brain_image_test);

%% Loading 1D Data

%{
For this operation the data sets should be in a folder called OneD_datain
a folder that is in the Path. Additionally the files should have a .txt
extension.
The phenotype information is contained in an xlsx file created from the tsv
the adhd200_preprocessed_phenotypics.tsv pheonotype file.
%}

OneD_names = dir('OneD_data/*.txt');
fMRI_data = cell(size(OneD_names,1),5);

fMRI_info = importdata('adhd200_preprocessed_phenotypics.xlsx');

load('entropy.mat')

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

features = zeros(size(OneD_names,1),10);

%% Matching Datasets with a Diagnosis
x = fMRI_info.data;

for j=1:size(fMRI_data,1)
    index = find(x(:,1) == fMRI_data{j,3});
    fMRI_data{j,1} = x(index,6);
end

for j=1:size(fMRI_data,1)
    if (fMRI_data{j,1})
        fMRI_data{j,4} = 1;
        class(j,1) = 1;
    else
        fMRI_data{j,4} = 0;
        class(j,1) = 0;
    end
end

%% Matching Datasets with a Site

x = fMRI_info.data;
site = zeros(size(class,1),1);

for j=1:size(fMRI_data,1)
    index = find(x(:,1) == fMRI_data{j,3});
    fMRI_data{j,5} = x(index,2);
    site(j,1) = x(index,2);
end

%% Ordering Based on Site

int1 = 1;
int2 = 1;
int3 = 1;
int4 = 1;
int5 = 1;
int6 = 1;
int7 = 1;
int8 = 1;

for i=1:size(class,1)
    if(site(i)==1)
        fMRI_data1(int1,:) = fMRI_data(i,:);
        int1 = int1 + 1;
    elseif(site(i)==2)
        fMRI_data2(int2,:) = fMRI_data(i,:);
        int2 = int2 + 1;
    elseif(site(i)==3)
        fMRI_data3(int3,:) = fMRI_data(i,:);
        int3 = int3 + 1;
    elseif(site(i)==4)
        fMRI_data4(int4,:) = fMRI_data(i,:);
        int4 = int4 + 1;
    elseif(site(i)==5)
        fMRI_data5(int5,:) = fMRI_data(i,:);
        int5 = int5 + 1;
    elseif(site(i)==6)
        fMRI_data6(int6,:) = fMRI_data(i,:);
        int6 = int6 + 1;
    elseif(site(i)==7)
        fMRI_data7(int7,:) = fMRI_data(i,:);
        int7 = int7 + 1;
    elseif(site(i)==8)
        fMRI_data8(int8,:) = fMRI_data(i,:);
        int8 = int8 + 1;
    else  
    end
end

fMRI_data1 = sortrows(fMRI_data1,1);
fMRI_data3 = sortrows(fMRI_data3,1);
fMRI_data4 = sortrows(fMRI_data4,1);
fMRI_data5 = sortrows(fMRI_data5,1);
fMRI_data6 = sortrows(fMRI_data6,1);
fMRI_data7 = sortrows(fMRI_data7,1);
fMRI_data8 = sortrows(fMRI_data8,1);

fMRI_data_Ordered = [fMRI_data1; fMRI_data3; fMRI_data4; fMRI_data5; fMRI_data6; fMRI_data7; fMRI_data8];

%% Matching Ordered Datasets with a Class

class_Ordered = zeros(size(class,1),1);
%{
for j=1:size(fMRI_data_Ordered,1)
    if (fMRI_data_Ordered{j,1} > 0)
        class_Ordered(j,1) = 1;
    else
        class_Ordered(j,1) = 0;
    end
end
%}

for j=1:size(fMRI_data_Ordered,1)
    class_Ordered(j,1) = fMRI_data_Ordered{j,1};
end

%% Matching Ordered Datasets with a Site

site_Ordered = zeros(size(class,1),1);

for j=1:size(fMRI_data_Ordered,1)
    site_Ordered(j,1) = fMRI_data_Ordered{j,5};
end

%% Plotting

x = 1:size(fMRI_data{1,2},1);
x =x';
temp_plot = fMRI_data{1,2};
temp_plot2 = fMRI_data{5,2};
temp_plot3 = fMRI_data{10,2};
temp_plot4 = fMRI_data{24,2};

dia = fMRI_data{1,1};
dia2 = fMRI_data{5,1};
dia3 = fMRI_data{10,1};
dia4 = fMRI_data{24,1};


for i=1:size(fMRI_data{1,2},2)
    figure(1)
    plot(x, temp_plot(:,i), '-o');
    title(num2str(dia))
    figure(2)
    plot(x, temp_plot2(:,i), '-o');
    title(num2str(dia2))
    figure(3)
    plot(x, temp_plot3(:,i), '-o');
    title(num2str(dia3))
    figure(4)
    plot(x, temp_plot4(:,i), '-o');
    title(num2str(dia4))
    pause
end

%% Max Power and Location of Max Power

clear pxx

for i=1:size(class,1)
    xx = fMRI_data{i,2};
    pxx = pwelch(xx(:,1));
    [p,l] = findpeaks(pxx(:,1), 'SortStr', 'descend');
    features(i,1) = p(1);
    features(i,2) = l(1);
end

%% Plotting for Max Power and Location of Max Power

for i=1:size(x,1)
    xx = x(:,i);
    xx2 = x2(:,i);
    pxx = pwelch(xx);
    pxx2 = pwelch(xx2);
    x_axis = 1:size(pxx,1);
    x_axis';
    figure(1)
    plot(x_axis, pxx);
    figure(2)
    plot(x_axis, pxx2);
    pause
end

%% FFT Power

clear fft_feat

for i=1:size(class,1)
    pxx = fMRI_data{i,2};
    fft = fftn(pxx(:,1));
    power_fft = pwelch(fft);
    [p,l] = findpeaks(power_fft(:,1), 'SortStr', 'descend');
    features(i,3) = p(1);
    features(i,4) = l(1);
end

%% Stats (Entropy: Mean, Median, Mode)

pxx = fMRI_data{i,2};
se_mean = zeros(size(class,1),size(pxx,2));
se_medain = zeros(size(class,1),size(pxx,2));
se_mode = zeros(size(class,1),size(pxx,2));

for i=1:size(class,1)
    pxx = fMRI_data{i,2};
    for j=1:size(pxx,2)
        se_temp = pentropy(pxx(:,j),1);
        se_mean(i,j) = mean(se_temp);
        se_medain(i,j) = median(se_temp);
        se_mode(i,j) = mode(se_temp);
    end
    i;
end

%% Plotting Site and Mean Entropy

figure(1)
x = 1:size(class,1);
subplot(3,1,1)
plot(x, se_mean(:,1))
subplot(3,1,2)
plot(x, site)
subplot(3,1,3)
plot(x,class)
 
%% Stats (Entropy: Mean, Median, Mode) with Site Sorted Data

clear se_mean_Ordered
clear se_median_Ordered
clear se_mode_Ordered

i = 1;

pxx = fMRI_data_Ordered{i,2};
se_mean_Ordered = zeros(size(class,1),size(pxx,2));
se_median_Ordered = zeros(size(class,1),size(pxx,2));
se_mode_Ordered = zeros(size(class,1),size(pxx,2));

for i=1:size(class,1)
    pxx = fMRI_data_Ordered{i,2};
    for j=1:1
        se_temp = pentropy(pxx(:,j),1);
        se_mean_Ordered(i,j) = mean(se_temp);
        se_median_Ordered(i,j) = median(se_temp);
        se_mode_Ordered(i,j) = mode(se_temp);
    end
    i
end


%% Median Filter
% Not sure how useful this is

med_filter = medfilt1(se_mean_Ordered(:,1));
plot(x, med_filter)

%% Plotting Site and Class Ordered Mean Entropy

figure(1)
x = 1:size(class,1);
subplot(3,1,1)
plot(x, se_mean_Ordered(:,1))
plot(x, med_filter)
title('Mean Entropy', 'FontSize', 16)
grid on;
grid minor;

subplot(3,1,2)
%plot(x, med_filter)
plot(x, site_Ordered)
title('Site', 'FontSize', 16)
grid on;

subplot(3,1,3)
plot(x,class_Ordered,'o')
title('Class', 'FontSize', 16)
grid on;

%% Scatter Plots Site 5

for i=1:size(se_median_Ordered5,2)
    for j=1:size(se_median_Ordered5,2)
        scatter(se_median_Ordered5(:,i),se_median_Ordered5(:,j), [], class_site5_Ordered)
        grid on;
        grid minor;
        pause
        j
    end
    i    
end

%% Stats (Entropy: Mean, Median, Mode) for Site 1 with Class Ordered Data

clear se_mean_Ordered1
clear se_median_Ordered1
clear se_mode_Ordered1

i = 1;

pxx = fMRI_data1{i,2};
se_mean_Ordered1 = zeros(size(fMRI_data1,1),size(pxx,2));
se_median_Ordered1 = zeros(size(fMRI_data1,1),size(pxx,2));
se_mode_Ordered1 = zeros(size(fMRI_data1,1),size(pxx,2));

for i=1:size(fMRI_data1,1)
    pxx = fMRI_data1{i,2};
    for j=1:10
        se_temp = pentropy(pxx(:,j),1);
        se_mean_Ordered1(i,j) = mean(se_temp);
        se_median_Ordered1(i,j) = median(se_temp);
        se_mode_Ordered1(i,j) = mode(se_temp);
    end
    i;
end

%% Stats (Entropy: Mean, Median, Mode) for Site 5 (NYU) with Class Ordered Data

clear se_mean_Ordered5
clear se_median_Ordered5
clear se_mode_Ordered5

i = 1;

pxx = fMRI_data5{i,2};
se_mean_Ordered5 = zeros(size(fMRI_data5,1),size(pxx,2));
se_median_Ordered5 = zeros(size(fMRI_data5,1),size(pxx,2));
se_mode_Ordered5 = zeros(size(fMRI_data5,1),size(pxx,2));

for i=1:size(fMRI_data5,1)
    pxx = fMRI_data5{i,2};
    for j=1:351
        se_temp = pentropy(pxx(:,j),1);
        se_mean_Ordered5(i,j) = mean(se_temp);
        se_median_Ordered5(i,j) = median(se_temp);
        se_mode_Ordered5(i,j) = mode(se_temp);
    end
    i
end

%% Plotting Site 5 (NYU) and Class Ordered Mean Entropy

class_site5_Ordered = fMRI_data5(:,1);
class_site5_Ordered = cell2mat(class_site5_Ordered);

figure(1)
x = 1:size(se_mean_Ordered5,1);
subplot(2,1,1)
plot(x, se_median_Ordered5(:,1))
grid on;
grid minor;

subplot(2,1,2)
plot(x,class_site5_Ordered,'o')
grid on;

%% Site 5 Zero Crossings

x = size(fMRI_data{1,2},2);
y = size(fMRI_data5{1,2},1);
z = size(fMRI_data5,1);

zero_cross = zeros(z,x);

for i=1:z
   for j=1:x
       zero_cross_temp = 0;
       temp = fMRI_data5(i,2);
       temp = cell2mat(temp);
       for k=1:(y-1)
           if(temp(k,j)*temp(k+1,j)<0)
               zero_cross_temp = zero_cross_temp + 1;  
           end
       end
       zero_cross(i,j) = zero_cross_temp;
   end
   i
end

%% Cross Correlation with Site 5 Data

for i=1:size(fMRI_data5{2,2},2)
a = fMRI_data5{i,2};
b = fMRI_data5{i+1,2};

corr = xcorr(a(:,1),b(:,1));

end

%% Sequential Classification
%{
Features from se_mean_Ordered 1 to 50 were tested. No real diserable value
was gained from this however the ordering was:[16;6;17;8;33;25;32;50;5;...
38;26;37;20;40;4;9;31;49;42;1;29;35;24;3;12;44;41;21;45;19;18;43;22;23;...
48;47;10;36;34;15;30;13;7;39;2;46;27;28;11;14]
%}


x = se_mean_Ordered(:,1:50);

y = class_Ordered;

c = cvpartition(y,'KFold',10);

fun = @(xtrain, ytrain, xtest, ytest) sum(ytest ~= classify(xtest, xtrain, ytrain));


[inmodel,history] = sequentialfs(fun, x, y, 'cv', c, 'Nfeatures', 50);
results = history.In(1,:);
final = find(results(1,:));
for p = 1:(length(history.In) - 1)
    results((p+1),:) = xor(history.In(p,:),history.In((p+1),:));
    final(p+1) = find(results(p+1,:));
end

final = final'; % transposed for better visual

%% Sequential Classification for Site 5
%{
Sequential classification of zero crossing for site 5 data with all 4
classes
[174;147;4;202;228;83;334;76;170;317;324;119;303;239;115;265;152;277;...
    172;151;127;293;243;276;311;347;2;160;74;281;320;177;146;15;80;81;...
    18;332;114;28;124;19;246;125;41;250;267;328;273;9;42;288;12;94;241;...
    24;222;183;138;52;191;92;230;39;57;49;8;68;69;73;123;135;44;16;118;...
    167;187;64;350;314;106;254;221;139;348;323;188;55;226;43;198;33;132;...
    78;161;122;214;194;321;32;253;71;219;178;168;22;193;236;335;255;120;...
    130;113;299;27;6;34;339;233;61;199;322;316;60;209;208;264;173;48;84;...
    156;333;268;128;305;196;53;153;229;107;159;307;298;100;25;36;45;313;...
    189;292;200;223;186;121;126;89;211;245;225;185;351;158;315;308;249;...
    5;47;66;275;336;251;58;287;140;90;10;117;349;337;210;204;62;201;242;...
    179;205;237;75;197;86;240;272;278;284;331;263;101;215;111;234;257;...
    279;171;163;103;23;247;51;116;131;248;195;65;235;29;180;143;259;330;...
    216;258;155;77;98;220;274;14;203;346;342;96;262;99;182;54;261;192;...
    82;63;244;30;297;269;218;46;50;150;154;217;148;67;232;175;291;93;...
    144;109;294;283;110;37;213;184;85;266;341;319;7;302;137;70;141;164;...
    325;231;56;252;59;26;290;329;38;104;280;343;1;309;136;327;17;256;...
    102;35;105;87;295;338;238;270;304;296;134;11;318;165;176;3;31;149;...
    21;79;157;91;326;260;142;207;300;166;306;301;162;181;95;289;212;344;...
    20;286;282;190;224;169;40;340;285;129;72;145;312;133;310;345;13;271;...
    88;108;206;97;227;112]
%}

%x = se_mode_Ordered5(:,1:15);
x = zero_cross(:,:);

y = class_site5_Ordered;

c = cvpartition(y,'KFold',10);

fun = @(xtrain, ytrain, xtest, ytest) sum(ytest ~= classify(xtest, xtrain, ytrain));


[inmodel,history] = sequentialfs(fun, x, y, 'cv', c, 'Nfeatures', 351);
results = history.In(1,:);
final = find(results(1,:));
for p = 1:(length(history.In) - 1)
    results((p+1),:) = xor(history.In(p,:),history.In((p+1),:));
    final(p+1) = find(results(p+1,:));
end

final = final'; % transposed for better visual

%% Generating a Classifier

LDA_power = fitcdiscr(se_mean_Ordered5(:,10),class_site5_Ordered,'CrossVal','on');

res = kfoldPredict(LDA_power);

%% Accuracy

clear acc

total = 0;

for i=1:size(class)
   if(res(i) == class(i))
       total = total + 1;
   end 
end

acc = total/size(class,1);

%% Accuracy site 5

clear acc
for b = 1:length(final)
    feats(b) = final(b,1);
    % test = se_mean_Ordered5(:,feats);
    LDA_power = fitcdiscr(zero_cross(:,feats),class_site5_Ordered,'CrossVal','on');

    res = kfoldPredict(LDA_power);

    total = 0;

    for i=1:size(class_site5_Ordered)
       if(res(i) == class_site5_Ordered(i))
           total = total + 1;
       end 
    end

    acc(b,1) = total/size(class_site5_Ordered,1);
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
clear fMRI_data1
clear fMRI_data2
clear fMRI_data3
clear fMRI_data4
%clear fMRI_data5
clear fMRI_data6
clear fMRI_data7
clear fMRI_data8
clear int1
clear int2
clear int3
clear int4
clear int5
clear int6
clear int7
clear int8

%% Lab 4 Plotting

figure(1)
x = 1:size(class,1);
plot(x, se_mean_Ordered(:,1))
grid on;
grid minor;

figure(2)
x = 1:size(class,1);
plot(x, se_median_Ordered(:,1))
grid on;
grid minor;

figure(3)
y = fMRI_data{1,2};
x = 1:size(y,1);
plot(x,y)
grid on;
grid minor;

figure(4)
y = fMRI_data{1,2};
x = 1:size(y,1);
plot(x,y(:,1))
grid on;
grid minor;

% Signal Peak Power 
figure(5)
x = 1:size(class,1);
plot(x, features(:,1))
ylim([0, 300]);
grid on;
grid minor;

% Signal Peak Power Locations
figure(6)
x = 1:size(class,1);
plot(x, features(:,2))
grid on;
grid minor;

% FFT Peak Power
figure(7)
x = 1:size(class,1);
plot(x, features(:,3))
ylim([0, 10000]);
grid on;
grid minor;

% FFT Peak Power Locations
figure(8)
x = 1:size(class,1);
plot(x, features(:,4))
grid on;
grid minor;

figure(9)
x = 1:size(se_mean_Ordered1,1);
plot(x, se_mean_Ordered1(:,1))
grid on;
grid minor;