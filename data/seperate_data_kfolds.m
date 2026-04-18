% 创建5折交叉验证下的，mlc_dataname_5folds.mat
% 要求：该文件下，分为train_ind, test_ind，分别存储的是
% 第x折，训练集（测试集）的样本索引号

clear;clc

dataname = 'mlc_CHD49';
load([dataname,'.mat'],'num');

% load(['data/',dataname,'_',num2str(training_data_ratio),'.mat'],'train_ind','test_ind');

% 需要存储的train_ind, test_ind
train_ind = cell(5, 1);
test_ind = cell(5, 1);


m = size(num, 1);
cv = cvpartition(m, 'kFold', 5);


for count = 1 : 5

    train_ifold_ind = linspace(1, length(cv.training(count)),length(cv.training(count)));
    test_ifold_ind = linspace(1, length(cv.test(count)),length(cv.test(count)));
    
    train_ind{count} = train_ifold_ind(cv.training(count));
    test_ind{count} = test_ifold_ind(cv.test(count));

end

save([dataname,'_5folds.mat'], 'train_ind', 'test_ind');