%matlab提供许多的built-in颜色图，每组颜色图都是64 * 3矩阵形式，例如：
%[0.0417   0     0]
%[0.0833   0     0]
%...
%[1.0     1.0   1.0]



clear all;
clc;

%计算程序运行时间.
tic;


%---------%only information here needs to be provided by user.----------%
saveDataPath = 'C:\FangCloudV2\Personal files\codes\matlab\saveMatlabColorMap\';
colormap = winter;
singleDataType = 'single';
%---------%only information here needs to be provided by user.----------%



%1. 保存该颜色图as .raw.
fid = fopen(strcat(saveDataPath, 'colormap_winter.raw'), 'w');
numOfMatrixElements = fwrite(fid, colormap, singleDataType);
fprintf('total number of matrix elements is: %d.\n', numOfMatrixElements);
fclose(fid);
fprintf('colormap.raw has been saved.\n');


%计算程序运行时间.
executionTime = toc;
fprintf('saveMatlabColormapAsRaw.m execution time = %fs.\n', executionTime);


