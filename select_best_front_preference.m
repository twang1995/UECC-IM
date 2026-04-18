function front_idx = select_best_front_preference(x_train, y_train, back_idx)
% ==============================================
% 功能：选择最优front_label（偏好马氏距离大+Gain Ratio小）
% 输入：x_train（特征矩阵，n×d）、y_train（候选标签矩阵，n×Q）、back_idx（back_label列索引）
% 输出：最优front_label的列索引（1≤front_idx≤Q，front_idx≠back_idx）
% ==============================================

%% 输入验证（鲁棒性保障）
[n_samples, n_features] = size(x_train);
[n_samples_y, n_labels] = size(y_train);
assert(n_samples == n_samples_y, 'x_train与y_train的样本数量不匹配');
assert(isscalar(back_idx) && back_idx >= 1 && back_idx <= n_labels && back_idx == round(back_idx), ...
    'back_idx必须是y_train中有效的列索引（正整数）');

% 提取back_label和候选front标签列
back_label = y_train(:, back_idx);
front_cols = setdiff(1:n_labels, back_idx);  % 排除back_label列
assert(~isempty(front_cols), '无候选front标签（y_train仅包含back_label列）');
n_candidates = n_labels - 1;

% 对x_train进行z-score标准化
x_train = zscore(x_train);


%% 评估每个候选front_label（马氏距离+Gain Ratio）
results = zeros(n_labels, 4);  % [马氏距离, GainRatio, 综合评分]
lambda = 1;                      % 惩罚系数（根据需求调整）

for i = 1:n_labels
    if i ~= back_idx
        current_front = y_train(:, i);

        % 计算马氏距离（越大越好）
        % y_train_temp = y_train;
        % y_train_temp(:, back_idx) = [];
        mahalanobis = compute_mahalanobis(x_train, back_label, current_front);

        % 计算准确的Gain Ratio（越小越好）
        gain_ratio = compute_gain_ratio(back_label, current_front);
        
        % 计算拆分后的均匀性，熵值（front label对back label=多数类拆分后，两个类别的熵值）
        H = cal_entropy_after_frontlabel(back_label, current_front);

        % 综合评分 = 马氏距离 - λ×Gain Ratio（偏好马氏距离大+Gain Ratio小）
        score = mahalanobis + lambda * gain_ratio;

        results(i, :) = [mahalanobis, H, score, gain_ratio];
    end
end


%% 选择最优front_label（综合评分最高）
results(:,1) = (results(:, 1) - min(results(:, 1))) ./ ( max(results(:,1)) - min(results(:,1)) );
results(:,2) = (results(:, 2) - min(results(:, 2))) ./ ( max(results(:,2)) - min(results(:,2)) );

lambda = 0.6;  % 0.5 OR 0.6
results(:,3) = lambda .* results(:,1) + (1 - lambda) .* results(:,2);

[~, best_idx] = max(results(:, 3));
front_idx = best_idx;

% **退回到原始UECC-IM的front label选择，即只依靠Gain Ratio最小来判别。
% 【目的】Ablation Study，原始UECC-IM下，只+Feature Selection部件
% [~, best_idx] = min(results(:, 4));
% front_idx = best_idx;


end

function mahalanobis = compute_mahalanobis(X, back_label, front_label)
% 校验标签是否为二值（0/1）
assert(all(ismember(unique(front_label), [0,1])) && all(ismember(unique(back_label), [0,1])), ...
       'front_label和back_label必须为二值标签（0或1）');

% 确定主类（多数类）
maj_front = 0;
if sum(front_label == 1) > sum(front_label == 0)
    maj_front = 1;
end
maj_back = 0;
if sum(back_label == 1) > sum(back_label == 0)
    maj_back = 1;
end

% 生成三类样本掩码（A: front主类&back主类；B: front非主类&back主类；C: back非主类）
A_mask = (front_label == maj_front) & (back_label == maj_back);
B_mask = (front_label ~= maj_front) & (back_label == maj_back);
C_mask = (back_label ~= maj_back);

XA = X(A_mask, :);  % A类特征矩阵
XB = X(B_mask, :);  % B类特征矩阵
XC = X(C_mask, :);  % C类特征矩阵

valid_dists = [];  % 存储有效距离

% 计算A与C的马氏距离（需A和C均有至少2个样本）
if size(XA, 1) >= 2 && size(XC, 1) >= 2
    mu_A = mean(XA);
    mu_C = mean(XC);
    Sigma_AC = cov([XA; XC]);       % 联合协方差矩阵
    Sigma_AC = Sigma_AC + 1e-6 * eye(size(Sigma_AC, 1));  % 正则化
    delta_AC = mu_A - mu_C;
    dist_AC = sqrt(delta_AC * inv(Sigma_AC) * delta_AC');
    valid_dists = [valid_dists, dist_AC];
end

% 计算B与C的马氏距离（需B和C均有至少2个样本）
if size(XB, 1) >= 2 && size(XC, 1) >= 2
    mu_B = mean(XB);
    mu_C = mean(XC);
    Sigma_BC = cov([XB; XC]);       % 联合协方差矩阵
    Sigma_BC = Sigma_BC + 1e-6 * eye(size(Sigma_BC, 1));  % 正则化
    delta_BC = mu_B - mu_C;
    dist_BC = sqrt(delta_BC * inv(Sigma_BC) * delta_BC');
    valid_dists = [valid_dists, dist_BC];
end

% 确定最终距离（根据需求选择min/max/mean）
if ~isempty(valid_dists)
    % mahalanobis = mean(valid_dists);  % 示例：取平均，反映整体可分性
    mahalanobis = min(valid_dists);  % 或取最小（最严格边界）
    % mahalanobis = max(valid_dists);  % 或取最大（最宽松边界）
else
    mahalanobis = 0;  % 无有效距离时返回0
end
end

% % ==============================================
% % 子函数1：计算马氏距离（与之前一致，已修复数值稳定性问题）
% % ==============================================
% function mahalanobis = compute_mahalanobis(X, back_label, front_label)
% % 划分三类样本：A(front=0&back=0), B(front=1&back=0), C(back=1)
% maj_value_front_label = -1;
% maj_value_back_label = -1;
% if sum(front_label == 0) > sum(front_label == 1)
%     maj_value_front_label = 0;
% else
%     maj_value_front_label = 1;
% end
% if sum(back_label == 0) > sum(back_label == 1)
%     maj_value_back_label = 0;
% else
%     maj_value_back_label = 1;
% end
% 
% A_mask = (front_label == maj_value_front_label) & (back_label == maj_value_back_label);
% B_mask = (front_label ~= maj_value_front_label) & (back_label == maj_value_back_label);
% C_mask = (back_label ~= maj_value_back_label);
% 
% XA = X(A_mask, :);  % A类特征矩阵
% XB = X(B_mask, :);  % B类特征矩阵
% XC = X(C_mask, :);  % C类特征矩阵
% mahalanobis = 0;
% valid_pairs = 0;
% 
% % 计算A类与C类的马氏距离（需A和C均有至少2个样本）
% if size(XA, 1) >= 2 && size(XC, 1) >= 2
%     mu_A = mean(XA);           % A类特征均值
%     mu_C = mean(XC);           % C类特征均值
%     Sigma_AC = cov([XA; XC]);  % 合并协方差矩阵（A和C的联合分布）
%     Sigma_AC = Sigma_AC + 1e-6 * eye(size(Sigma_AC, 1));  % 正则化（防奇异）
%     delta_AC = mu_A - mu_C;    % 均值差向量
%     dist_AC = sqrt(delta_AC * inv(Sigma_AC) * delta_AC');  % 马氏距离公式
%     mahalanobis = mahalanobis + dist_AC;
%     valid_pairs = valid_pairs + 1;
% else
%     dist_AC = 0;
% end
% 
% % 计算B类与C类的马氏距离（需B和C均有至少2个样本）
% if size(XB, 1) >= 2 && size(XC, 1) >= 2
%     mu_B = mean(XB);           % B类特征均值
%     mu_C = mean(XC);           % C类特征均值
%     Sigma_BC = cov([XB; XC]);  % 合并协方差矩阵（B和C的联合分布）
%     Sigma_BC = Sigma_BC + 1e-6 * eye(size(Sigma_BC, 1));  % 正则化
%     delta_BC = mu_B - mu_C;    % 均值差向量
%     dist_BC = sqrt(delta_BC * inv(Sigma_BC) * delta_BC');  % 马氏距离公式
%     mahalanobis = mahalanobis + dist_BC;
%     valid_pairs = valid_pairs + 1;
% else
%      dist_BC = 0;
% end
% 
% mahalanobis = min([dist_BC, dist_AC]);
% 
% % 处理无有效距离的情况（返回0）
% % if valid_pairs > 0
% %     mahalanobis = mahalanobis / valid_pairs;
% % else
% %     mahalanobis = 0;
% % end
% end


% ==============================================
% 子函数2：计算准确的Gain Ratio（修复符号错误+分裂信息）
% ==============================================
function gain_ratio = compute_gain_ratio(target, feature)
% 输入：target为目标变量（back_label，列向量），feature为特征变量（front_label，列向量）
% 输出：gain_ratio为信息增益率（非负数，范围[0,1]）

%% 步骤1：计算目标变量的经验熵 H(target)
n_total = numel(target);                  % 总样本数
target_vals = unique(target);             % 目标变量的唯一取值
H_target = 0;                             % 目标变量的熵

for i = 1:length(target_vals)
    cnt = sum(target == target_vals(i));  % 目标取值为target_vals(i)的样本数
    p = cnt / n_total;                    % 该取值的概率
    H_target = H_target - p * log2(p + eps);  % 修正符号：添加负号
end


%% 步骤2：计算条件熵 H(target | feature)
feature_vals = unique(feature);           % 特征变量的唯一取值
H_cond = 0;                               % 条件熵

for j = 1:length(feature_vals)
    % 提取特征等于feature_vals(j)的样本索引
    idx = find(feature == feature_vals(j));
    cnt_feature = length(idx);            % 该特征值的样本数
    p_feature = cnt_feature / n_total;    % 该特征值的概率

    % 计算目标变量在该特征值下的经验熵 H(target | feature=feature_vals(j))
    target_subset = target(idx);
    H_subset = 0;
    for k = 1:length(target_vals)
        cnt_subset = sum(target_subset == target_vals(k));  % 目标取值为target_vals(k)的样本数
        p_subset = cnt_subset / (cnt_feature + eps);        % 避免除以0
        H_subset = H_subset - p_subset * log2(p_subset + eps);  % 修正符号
    end

    % 累加条件熵项：p(feature=feature_vals(j)) * H(target | feature=feature_vals(j))
    H_cond = H_cond + p_feature * H_subset;
end


%% 步骤3：计算信息增益（Information Gain）
info_gain = H_target - H_cond;


%% 步骤4：计算分裂信息（Split Information，特征自身的熵）
split_info = 0;
for j = 1:length(feature_vals)
    cnt_feature = sum(feature == feature_vals(j));  % 特征取值为feature_vals(j)的样本数
    p_split = cnt_feature / n_total;                 % 该特征值的概率
    split_info = split_info - p_split * log2(p_split + eps);  % 修正符号
end


%% 步骤5：计算Gain Ratio（避免除零错误）
if split_info < eps
    gain_ratio = 0;  % 特征无变化时，增益率为0
else
    gain_ratio = info_gain / split_info;
    gain_ratio = max(gain_ratio, 0);  % 确保非负
    gain_ratio = min(gain_ratio, 1);  % 理论最大值为H_target（通常≤1）
end
end

%% 计算front label对back label拆分后，熵值
% H = -p1log(p1) - p2log(p2), p1 = N1 / N0, p2 = N2 / N0
% H \in [0, log2]，值越大越均匀
function H = cal_entropy_after_frontlabel(back_label, front_label)
% 输入：target为目标变量（back_label，列向量），feature为特征变量（front_label，列向量）
% 输出：gini_value为基尼系数（非负数，范围[0,0.5]）
maj_value_front_label = -1;
maj_value_back_label = -1;
if sum(front_label == 0) > sum(front_label == 1)
    maj_value_front_label = 0;
else
    maj_value_front_label = 1;
end
if sum(back_label == 0) > sum(back_label == 1)
    maj_value_back_label = 0;
else
    maj_value_back_label = 1;
end

N0 = sum(back_label == maj_value_back_label);
N1 = sum(back_label == maj_value_back_label & front_label ~= maj_value_front_label); % 10
N2 = sum(back_label == maj_value_back_label & front_label == maj_value_front_label); % 00
if N1 + N2 ~= N0, wiaoeuiwoau; end
p1 = N1 / N0;
p2 = N2 / N0;
H = - p1 * log2(p1) - p2 * log2(p2);

end