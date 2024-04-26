% 设置CSV文件所在文件夹路径
csvFolderPath = 'E:\数据\一维\train\1d\微震\';
% 获取文件夹中所有CSV文件的列表
csvFiles = dir(fullfile(csvFolderPath, '*.csv'));
% 创建一个MATLAB结构来存储所有数据
dataStruct = struct('train_X', [], 'train_Y', [],'test_X', [], 'test_Y', [],'valid_X', [], 'valid_Y', []);
% 循环处理每个CSV文件
for i = 1:length(csvFiles)
    i
    % 构建CSV文件的完整路径
    csvFilePath = fullfile(csvFolderPath, csvFiles(i).name);
    % 读取CSV文件
    data = readtable(csvFilePath);
    % 提取第二列数据
    data_matrix = table2array(data); % 将表格数据转换为数值矩阵
    second_column = data_matrix(:, 2)'; % 提取第二列数据
    % 标签赋值为0

    % 存储数据和标签到MATLAB结构
    dataStruct.train_X = [dataStruct.train_X;second_column];
    dataStruct.train_Y = [dataStruct.train_Y; 1 0];
end

newCsvFolderPath = 'E:\数据\一维\train\1d\爆破\';

newCsvFiles = dir(fullfile(newCsvFolderPath, '*.csv'));

newdataStruct = struct('train_X', [], 'train_Y', []);

for i = 1:length(newCsvFiles)

    newCsvFilePath = fullfile(newCsvFolderPath, newCsvFiles(i).name);
    
    newCsvData = readtable(newCsvFilePath);
    
    newdata_matrix = table2array(newCsvData); % 将表格数据转换为数值矩阵
    newSecondColumn = newdata_matrix(:, 2)'; 

    newdataStruct.train_X = [newdataStruct.train_X; newSecondColumn ]; % 使用分号表示逐行添加数据

    newdataStruct.train_Y = [newdataStruct.train_Y; 0 1]; % 使用分号表示逐行添加标签
end
% 继续使用之前的MATLAB结构（假设之前的数据存储在dataStruct中）
% 将新数据与之前的数据合并
dataStruct.train_X = [dataStruct.train_X; newdataStruct.train_X];
dataStruct.train_Y = [dataStruct.train_Y; newdataStruct.train_Y];


% 设置CSV文件所在文件夹路径
testPath = 'E:\数据\一维\test\1d\微震\';
% 获取文件夹中所有CSV文件的列表
testcsvFiles = dir(fullfile(testPath, '*.csv'));
% 循环处理每个CSV文件
for i = 1:length(testcsvFiles)
    i
    % 构建CSV文件的完整路径
    testcsvFilePath = fullfile(testPath, testcsvFiles(i).name);
    % 读取CSV文件
    testdata = readtable(testcsvFilePath);
    % 提取第二列数据
    testdata_matrix = table2array(testdata); % 将表格数据转换为数值矩阵
    testsecond_column = testdata_matrix(:, 2)'; % 提取第二列数据
    % 标签赋值为0

    % 存储数据和标签到MATLAB结构
    dataStruct.test_X = [dataStruct.test_X;testsecond_column];
    dataStruct.test_Y = [dataStruct.test_Y; 1 0];
end

newCsvFolderPath1 = 'E:\数据\一维\train\1d\爆破\';

newCsvFiles1 = dir(fullfile(newCsvFolderPath1, '*.csv'));

newdataStruct1 = struct('test_X', [], 'test_Y', []);

for i = 1:length(newCsvFiles1)
    i
    newCsvFilePath1 = fullfile(newCsvFolderPath1, newCsvFiles1(i).name);
    
    newCsvData1 = readtable(newCsvFilePath1);
    
    newdata_matrix1 = table2array(newCsvData1); % 将表格数据转换为数值矩阵
    newSecondColumn1 = newdata_matrix1(:, 2)'; 

    newdataStruct1.test_X = [newdataStruct1.test_X; newSecondColumn1 ]; % 使用分号表示逐行添加数据

    newdataStruct1.test_Y = [newdataStruct1.test_Y; 0 1]; % 使用分号表示逐行添加标签
end
% 继续使用之前的MATLAB结构（假设之前的数据存储在dataStruct中）
% 将新数据与之前的数据合并



dataStruct.test_X = [dataStruct.test_X; newdataStruct1.test_X];
dataStruct.test_Y = [dataStruct.test_Y; newdataStruct1.test_Y];

% 设置CSV文件所在文件夹路径
validPath = 'E:\数据\一维\valid\1d\微震\';
% 获取文件夹中所有CSV文件的列表
validcsvFiles = dir(fullfile(validPath, '*.csv'));
% 循环处理每个CSV文件
for i = 1:length(validcsvFiles)
    i
    % 构建CSV文件的完整路径
    validcsvFilePath = fullfile(validPath, validcsvFiles(i).name);
    % 读取CSV文件
    validdata = readtable(validcsvFilePath);
    % 提取第二列数据
    validdata_matrix = table2array(validdata); % 将表格数据转换为数值矩阵
    validsecond_column = validdata_matrix(:, 2)'; % 提取第二列数据
    % 标签赋值为0
    % 存储数据和标签到MATLAB结构
    dataStruct.valid_X = [dataStruct.valid_X;validsecond_column];
    dataStruct.valid_Y = [dataStruct.valid_Y; 1 0];
end

newCsvFolderPath2 = 'E:\数据\一维\valid\1d\爆破\';

newCsvFiles2 = dir(fullfile(newCsvFolderPath2, '*.csv'));

newdataStruct2 = struct('valid_X', [], 'valid_Y', []);

for i = 1:length(newCsvFiles2)
    i

    newCsvFilePath2 = fullfile(newCsvFolderPath2, newCsvFiles2(i).name);
    
    newCsvData2 = readtable(newCsvFilePath2);
    
    newdata_matrix2 = table2array(newCsvData2); % 将表格数据转换为数值矩阵
    newSecondColumn2 = newdata_matrix2(:, 2)'; 

    newdataStruct2.valid_X = [newdataStruct2.valid_X; newSecondColumn2 ]; % 使用分号表示逐行添加数据

    newdataStruct2.valid_Y = [newdataStruct2.valid_Y; 0 1]; % 使用分号表示逐行添加标签
end
% 继续使用之前的MATLAB结构（假设之前的数据存储在dataStruct中）
% 将新数据与之前的数据合并
dataStruct.valid_X = [dataStruct.valid_X; newdataStruct2.valid_X];
dataStruct.valid_Y = [dataStruct.valid_Y; newdataStruct2.valid_Y];

save('microseism_combined.mat', '-struct', 'dataStruct');


