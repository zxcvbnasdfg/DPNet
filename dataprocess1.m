% ����CSV�ļ������ļ���·��
csvFolderPath = 'E:\����\һά\train\1d\΢��\';
% ��ȡ�ļ���������CSV�ļ����б�
csvFiles = dir(fullfile(csvFolderPath, '*.csv'));
% ����һ��MATLAB�ṹ���洢��������
dataStruct = struct('train_X', [], 'train_Y', [],'test_X', [], 'test_Y', [],'valid_X', [], 'valid_Y', []);
% ѭ������ÿ��CSV�ļ�
for i = 1:length(csvFiles)
    i
    % ����CSV�ļ�������·��
    csvFilePath = fullfile(csvFolderPath, csvFiles(i).name);
    % ��ȡCSV�ļ�
    data = readtable(csvFilePath);
    % ��ȡ�ڶ�������
    data_matrix = table2array(data); % ���������ת��Ϊ��ֵ����
    second_column = data_matrix(:, 2)'; % ��ȡ�ڶ�������
    % ��ǩ��ֵΪ0

    % �洢���ݺͱ�ǩ��MATLAB�ṹ
    dataStruct.train_X = [dataStruct.train_X;second_column];
    dataStruct.train_Y = [dataStruct.train_Y; 1 0];
end

newCsvFolderPath = 'E:\����\һά\train\1d\����\';

newCsvFiles = dir(fullfile(newCsvFolderPath, '*.csv'));

newdataStruct = struct('train_X', [], 'train_Y', []);

for i = 1:length(newCsvFiles)

    newCsvFilePath = fullfile(newCsvFolderPath, newCsvFiles(i).name);
    
    newCsvData = readtable(newCsvFilePath);
    
    newdata_matrix = table2array(newCsvData); % ���������ת��Ϊ��ֵ����
    newSecondColumn = newdata_matrix(:, 2)'; 

    newdataStruct.train_X = [newdataStruct.train_X; newSecondColumn ]; % ʹ�÷ֺű�ʾ�����������

    newdataStruct.train_Y = [newdataStruct.train_Y; 0 1]; % ʹ�÷ֺű�ʾ������ӱ�ǩ
end
% ����ʹ��֮ǰ��MATLAB�ṹ������֮ǰ�����ݴ洢��dataStruct�У�
% ����������֮ǰ�����ݺϲ�
dataStruct.train_X = [dataStruct.train_X; newdataStruct.train_X];
dataStruct.train_Y = [dataStruct.train_Y; newdataStruct.train_Y];


% ����CSV�ļ������ļ���·��
testPath = 'E:\����\һά\test\1d\΢��\';
% ��ȡ�ļ���������CSV�ļ����б�
testcsvFiles = dir(fullfile(testPath, '*.csv'));
% ѭ������ÿ��CSV�ļ�
for i = 1:length(testcsvFiles)
    i
    % ����CSV�ļ�������·��
    testcsvFilePath = fullfile(testPath, testcsvFiles(i).name);
    % ��ȡCSV�ļ�
    testdata = readtable(testcsvFilePath);
    % ��ȡ�ڶ�������
    testdata_matrix = table2array(testdata); % ���������ת��Ϊ��ֵ����
    testsecond_column = testdata_matrix(:, 2)'; % ��ȡ�ڶ�������
    % ��ǩ��ֵΪ0

    % �洢���ݺͱ�ǩ��MATLAB�ṹ
    dataStruct.test_X = [dataStruct.test_X;testsecond_column];
    dataStruct.test_Y = [dataStruct.test_Y; 1 0];
end

newCsvFolderPath1 = 'E:\����\һά\train\1d\����\';

newCsvFiles1 = dir(fullfile(newCsvFolderPath1, '*.csv'));

newdataStruct1 = struct('test_X', [], 'test_Y', []);

for i = 1:length(newCsvFiles1)
    i
    newCsvFilePath1 = fullfile(newCsvFolderPath1, newCsvFiles1(i).name);
    
    newCsvData1 = readtable(newCsvFilePath1);
    
    newdata_matrix1 = table2array(newCsvData1); % ���������ת��Ϊ��ֵ����
    newSecondColumn1 = newdata_matrix1(:, 2)'; 

    newdataStruct1.test_X = [newdataStruct1.test_X; newSecondColumn1 ]; % ʹ�÷ֺű�ʾ�����������

    newdataStruct1.test_Y = [newdataStruct1.test_Y; 0 1]; % ʹ�÷ֺű�ʾ������ӱ�ǩ
end
% ����ʹ��֮ǰ��MATLAB�ṹ������֮ǰ�����ݴ洢��dataStruct�У�
% ����������֮ǰ�����ݺϲ�



dataStruct.test_X = [dataStruct.test_X; newdataStruct1.test_X];
dataStruct.test_Y = [dataStruct.test_Y; newdataStruct1.test_Y];

% ����CSV�ļ������ļ���·��
validPath = 'E:\����\һά\valid\1d\΢��\';
% ��ȡ�ļ���������CSV�ļ����б�
validcsvFiles = dir(fullfile(validPath, '*.csv'));
% ѭ������ÿ��CSV�ļ�
for i = 1:length(validcsvFiles)
    i
    % ����CSV�ļ�������·��
    validcsvFilePath = fullfile(validPath, validcsvFiles(i).name);
    % ��ȡCSV�ļ�
    validdata = readtable(validcsvFilePath);
    % ��ȡ�ڶ�������
    validdata_matrix = table2array(validdata); % ���������ת��Ϊ��ֵ����
    validsecond_column = validdata_matrix(:, 2)'; % ��ȡ�ڶ�������
    % ��ǩ��ֵΪ0
    % �洢���ݺͱ�ǩ��MATLAB�ṹ
    dataStruct.valid_X = [dataStruct.valid_X;validsecond_column];
    dataStruct.valid_Y = [dataStruct.valid_Y; 1 0];
end

newCsvFolderPath2 = 'E:\����\һά\valid\1d\����\';

newCsvFiles2 = dir(fullfile(newCsvFolderPath2, '*.csv'));

newdataStruct2 = struct('valid_X', [], 'valid_Y', []);

for i = 1:length(newCsvFiles2)
    i

    newCsvFilePath2 = fullfile(newCsvFolderPath2, newCsvFiles2(i).name);
    
    newCsvData2 = readtable(newCsvFilePath2);
    
    newdata_matrix2 = table2array(newCsvData2); % ���������ת��Ϊ��ֵ����
    newSecondColumn2 = newdata_matrix2(:, 2)'; 

    newdataStruct2.valid_X = [newdataStruct2.valid_X; newSecondColumn2 ]; % ʹ�÷ֺű�ʾ�����������

    newdataStruct2.valid_Y = [newdataStruct2.valid_Y; 0 1]; % ʹ�÷ֺű�ʾ������ӱ�ǩ
end
% ����ʹ��֮ǰ��MATLAB�ṹ������֮ǰ�����ݴ洢��dataStruct�У�
% ����������֮ǰ�����ݺϲ�
dataStruct.valid_X = [dataStruct.valid_X; newdataStruct2.valid_X];
dataStruct.valid_Y = [dataStruct.valid_Y; newdataStruct2.valid_Y];

save('microseism_combined.mat', '-struct', 'dataStruct');


