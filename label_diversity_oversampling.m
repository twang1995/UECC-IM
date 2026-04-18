function ind_selected_each_bin = label_diversity_oversampling(X_train, y_train, q, alphabet, number)
    % 提取当前 bin 中的样本特征和标签
    X_bin = X_train(alphabet, :);
    y_bin = y_train(alphabet, :);
    num_samples = size(X_bin, 1);

    % 对特征进行归一化处理
    X_bin_normalized = normalize(X_bin, 'range');

    % 移除当前学习的标签
    other_labels = y_bin(:, [1:q - 1, q + 1:end]);

    % 初始化多样性得分向量
    diversity_score = zeros(num_samples, 1);

    % 计算每个样本与其他样本的平均距离（结合特征和标签）
    for i = 1:num_samples
        feature_distances = pdist2(X_bin_normalized(i, :), X_bin_normalized, 'euclidean');
        label_distances = zeros(num_samples, 1);
        for j = 1:num_samples
            intersection = sum(other_labels(i, :) == other_labels(j, :) & other_labels(i, :) ~= 0);
            union = sum(other_labels(i, :) ~= 0 | other_labels(j, :) ~= 0);
            if union == 0
                label_distances(j) = 0;
            else
                label_distances(j) = 1 - (intersection / union);
            end
        end

        % 结合特征距离和标签距离（这里简单平均，可根据实际调整权重）
        %（特征距离也保持在0～1）
        feature_distances = feature_distances ./ (max(feature_distances) - min(feature_distances));
        % if sum(label_distances) ~= 0
        %     label_distances = label_distances ./ (max(label_distances) - min(label_distances));
        % end

        combined_distances = (feature_distances + label_distances') / 2;
        diversity_score(i) = mean(combined_distances);
    end

    % 根据多样性得分进行排序
    [~, sorted_indices] = sort(diversity_score, 'descend');

    % 选取样本索引
    if number <= num_samples
        ind_selected_each_bin = alphabet(sorted_indices(1:number));
    else
        % 先选取所有现有样本
        ind_selected_each_bin = alphabet(sorted_indices);
        remaining = number - num_samples;
        % 确保 remaining 不超过 sorted_indices 的长度
        remaining = min(remaining, num_samples);
        % 再重复选取前几个样本
        additional_indices = sorted_indices(1:remaining);
        ind_selected_each_bin = [ind_selected_each_bin, alphabet(additional_indices)];
    end
end    