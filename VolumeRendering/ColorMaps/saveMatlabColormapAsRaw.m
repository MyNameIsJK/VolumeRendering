%matlab�ṩ����built-in��ɫͼ��ÿ����ɫͼ����64 * 3������ʽ�����磺
%[0.0417   0     0]
%[0.0833   0     0]
%...
%[1.0     1.0   1.0]



clear all;
clc;

%�����������ʱ��.
tic;


%---------%only information here needs to be provided by user.----------%
saveDataPath = 'C:\FangCloudV2\Personal files\codes\matlab\saveMatlabColorMap\';
colormap = winter;
singleDataType = 'single';
%---------%only information here needs to be provided by user.----------%



%1. �������ɫͼas .raw.
fid = fopen(strcat(saveDataPath, 'colormap_winter.raw'), 'w');
numOfMatrixElements = fwrite(fid, colormap, singleDataType);
fprintf('total number of matrix elements is: %d.\n', numOfMatrixElements);
fclose(fid);
fprintf('colormap.raw has been saved.\n');


%�����������ʱ��.
executionTime = toc;
fprintf('saveMatlabColormapAsRaw.m execution time = %fs.\n', executionTime);


