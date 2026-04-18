% ==================== 用户输入区域 ====================
% 替换为您的自变量矩阵（m×n，m样本，n变量）

dataname = 'mlc_flags';
load([dataname,'.mat'],'num');
addpath('GAINRATIO');

%flags
x = num(:,1:19);
y = num(:,20:end);

% emotions
% x = num(:,1:72);
% y = num(:,73:end);


X = num;  % 示例数据（6个自变量，500样本）

q = size(y, 2);

gain_ratio_matrix = zeros(q,q);

for i = 1 : q
    for j = 1 : q
        gain_ratio_matrix(i, j) = cal_gain_ratio(y(:, i), y(:, j));
    end
end

