function [y_predict3_t, pr_predict3_t] = Predict_3classes(classifier_3_1, classifier_3_2, x_t)
%PREDICT_3CLASSES 人工三分类的预测函数
[~, pr_3_1_t] = BaseClassifierPredict(x_t, classifier_3_1);
[~, pr_3_2_t] = BaseClassifierPredict(x_t, classifier_3_2);

% beta, alpha均等于1
beta = 1; alpha = 1;



% if classifier_3_1.Label(1) == 1 && classifier_3_2.Label(1) == 1

    y_predict3_t = -1 .* ones(size(x_t, 1), 1);
    y_predict3_t([find(pr_3_1_t(:,1) + pr_3_2_t(:,1) > 1)]) = 1;
    pr_1 = (beta .* pr_3_1_t(:,1) + alpha .* pr_3_2_t(:,1)) .* 0.5;
    pr_predict3_t = [pr_1, 1 - pr_1];
% else
%     wioaueoirhao
% end



end

