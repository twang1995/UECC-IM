function [sorted_labels, entropy_values] = conditional_entropy_sorting(Y)
    % 输入：Y - 多标签数据矩阵（n×L，n为样本数，L为标签数，元素为0或1）
    % 输出：sorted_labels - 按条件熵升序排列的标签索引（1~L）
    %       entropy_values - 每个标签的条件熵值
    
    % 输入检查
    if ~ismatrix(Y) || ~all(Y(:) == 0 | Y(:) == 1)
        error('输入Y必须是0-1逻辑矩阵（n×L）');
    end
    [n, L] = size(Y);
    if L < 2
        error('至少需要2个标签才能计算条件熵');
    end
    
    % 初始化条件熵数组
    entropy_values = zeros(1, L);
    
    % 遍历每个标签，计算其条件熵 H(Y_i | Y_{-i})
    for i = 1:L
        % 提取除第i个标签外的其他标签（Y_{-i}）
        Y_rest = Y(:, [1:i-1, i+1:L]);
        
        % 将Y_rest的每行转换为唯一字符串（作为条件状态的标识）
        % 示例：[1 0 1] → '101'，用于统计相同状态的样本数
        state_keys = cell(n, 1);
        for j = 1:n
            state_keys{j} = sprintf('%d', Y_rest(j, :));  % 转换为字符串（如'101'）
        end
        
        % 统计每个状态下Y_i的0/1计数
        unique_states = unique(state_keys);  % 所有唯一条件状态
        num_states = length(unique_states);
        count_0 = zeros(num_states, 1);  % 状态s下Y_i=0的样本数
        count_1 = zeros(num_states, 1);  % 状态s下Y_i=1的样本数
        
        for s = 1:num_states
            idx = strcmp(state_keys, unique_states{s});  % 该状态对应的样本索引
            y_i_values = Y(idx, i);                      % 该状态下Y_i的取值
            count_0(s) = sum(y_i_values == 0);
            count_1(s) = sum(y_i_values == 1);
        end
        
        % 计算条件熵 H(Y_i | Y_{-i})
        total_entropy = 0;
        for s = 1:num_states
            total_samples_in_state = count_0(s) + count_1(s);
            if total_samples_in_state == 0
                continue;  % 无样本的状态不贡献熵
            end
            
            % 状态s出现的概率 P(s) = 样本数 / 总样本数n
            P_s = total_samples_in_state / n;
            
            % 条件概率 P(Y_i=0|s) 和 P(Y_i=1|s)
            p0 = count_0(s) / total_samples_in_state;
            p1 = count_1(s) / total_samples_in_state;
            
            % 避免log2(0)的情况（当p0或p1为0时，对应项为0）
            entropy_s = 0;
            if p0 > 0
                entropy_s = entropy_s - p0 * log2(p0);
            end
            if p1 > 0
                entropy_s = entropy_s - p1 * log2(p1);
            end
            
            % 累加状态s对总条件熵的贡献（P(s) * entropy_s）
            total_entropy = total_entropy + P_s * entropy_s;
        end
        
        entropy_values(i) = total_entropy;
    end
    
    % 按条件熵升序排列标签索引（熵越低越靠前）
    [~, sorted_indices] = sort(entropy_values);
    sorted_labels = sorted_indices;
end
    