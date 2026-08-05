function [IRLbl_lambda] = IRLbl(y, lambda)
%Cal_IRLbl 计算单个标签的不平衡比
%   单个标签L_lambda，形参lambda是第几个
%针对标签\lambda，多数标签与标签\lambda之间的比例。
% 最频繁的标签，IRLbl=1；
% 其余标签，IRLbl值更大。
% 该值越大，对应标签的不平衡程度越高。

%样本数量m，标签数量q
m = size(y, 1);
q = size(y, 2);

% Step1 计算最大的 max(  \sum_{i=1}^m h(lambda', Y_i)  )，最大值存储到h_max
%   其中，h(lambda, Y_i)是指第i个样本是否属于L_lambda，属于为1，否则为0
h_max = max(sum(y == 1));

% Step2 统计当前标签lambda在所有样本中出现的次数 \sum_{i=1}^m h(lambda, Y_i)
h_lambda = sum(y(:, lambda) == 1);

% Step3 计算IRLBl 第lambda个标签的不平衡比
IRLbl_lambda = h_max / h_lambda;




end

