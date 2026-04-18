function [classifier_3_1, classifier_3_2] = Train_3classes(y_train_cur, x_train_cur, back_label, front_label)
%TRAIN_3CLASSES 进行单个标签的三分类训练
    % 寻找front_label
    q = size(y_train_cur, 2);

    % gain_ratio_arr = zeros(1, q);
    % for lambda_front = 1 : q
    %     if lambda_front ~= back_label
    %         gain_ratio_arr(1, lambda_front) = Ecents([y_train_cur(:,lambda_front),y_train_cur(:,back_label)]) / expEnt(y_train_cur(:, back_label));
    %     end
    % end
    % gain_ratio_arr(back_label) = 100;
    % 
    % [~, front_label] = min(gain_ratio_arr);




    if sum(y_train_cur(:, back_label) == -1) >= sum(y_train_cur(:, back_label) == 1)
        maj_value_back_label = -1;
%         min_samples_num = sum(y_train(:, back_label) == 1);
    else
        maj_value_back_label = 1;
%         min_samples_num = sum(y_train(:, back_label) == -1);
    end

    if sum(y_train_cur(:, front_label) == -1) >= sum(y_train_cur(:, front_label) == 1)
        maj_value_front_label = -1;
        %         min_samples_num = sum(y_train(:, back_label) == 1);
    else
        maj_value_front_label = 1;
        %         min_samples_num = sum(y_train(:, back_label) == -1);
    end
    
%     maj_value_front_label = -1;
%     maj_value_back_label = -1;


    
    
    % 样本划分
    class_00_index = find(y_train_cur(:,front_label) == maj_value_front_label & y_train_cur(:,back_label) == maj_value_back_label)';
    class_10_index = find(y_train_cur(:,front_label) ~= maj_value_front_label & y_train_cur(:,back_label) == maj_value_back_label)';
    class_1_index = find(y_train_cur(:,back_label) ~= maj_value_back_label)';

    % 00 vs. _1
        % 数据选择
    y_train_3_1 = y_train_cur;
    y_train_3_1 = y_train_3_1([class_00_index, class_1_index], :);
    x_train_3_1 = x_train_cur;
    x_train_3_1 = x_train_3_1([class_00_index, class_1_index], :);
        % modeling
    classifier_3_1 = BaseClassifierTrain(y_train_3_1(:, back_label), x_train_3_1);

    % 10 vs. _1
        % 数据选择
    y_train_3_2 = y_train_cur;
    y_train_3_2 = y_train_3_2([class_10_index, class_1_index], :);
    x_train_3_2 = x_train_cur;
    x_train_3_2 = x_train_3_2([class_10_index, class_1_index], :);
        % modeling
    classifier_3_2 = BaseClassifierTrain(y_train_3_2(:, back_label), x_train_3_2);
    
    
end

