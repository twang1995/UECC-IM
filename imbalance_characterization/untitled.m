addpath('datasets');
addpath('liblinear');
dataname = 'mlc_flags';
%dataname_x = 'CAL500x';
%dataname_y = 'CAL500y';
training_data_ratio = 0.8;  % split a part of data for training, the others for testing
repeat_time = 1;  % times of random splits
T = 1;  % number of circles in CCE
addpath('/Users/wtl/Desktop/CC新探索/ART_CC_without_features');
load([dataname,'.mat'],'num');

%emotions
% x = num(:,1:72);
% y = num(:,73:end);

%yeast
% x = num(:,1:103);
% y = num(:,104:end);

%flags
x = num(:,1:19);
y = num(:,20:end);
y(y(:,1)==1,1) = 2;
y(y(:,1)==0,1) = 1;
y(y(:,1)==2,1) = 0;

y(y(:,5)==1,5) = 2;
y(y(:,5)==0,5) = 1;
y(y(:,5)==2,5) = 0;

%enron
% x = num(:,1:1001);
% y = num(:,1002:end);

%medical
% x = num(:,1:1449);
% y = num(:,1450:end);



y(y==0) = -1;
x = sparse(x);
load([dataname,'_',num2str(training_data_ratio),'.mat'],'train_ind','test_ind');
echo = 1;
x_train = x(train_ind{echo},:);
y_train = y(train_ind{echo},:);



%====================================================================================================

% front_label = 1;
% back_label = 6;


q = size(y, 2);
scumble_matrix = zeros(q,q);
for front_label = 1 : q
    for back_label = 1 : q
%         scumble_matrix(front_label, back_label) = scumble(y(:, [front_label, back_label]));
        scumble_matrix(front_label, back_label) = Copy_of_scumble_between_labels(y, front_label, back_label);
    end
end



% if front_label ~= back_label
%     %00，10，_1的索引
%     indexes00 = intersect(find(y_train(:,front_label) == -1), find(y_train(:,back_label) == -1));
%     indexes10 = intersect(find(y_train(:,front_label) == 1), find(y_train(:,back_label) == -1));
%     indexes_1 = find(y_train(:,back_label) == 1);
% 
%     % 二分类器
%     classifier_2 = BaseClassifierTrain(y_train(:,back_label), x_train);
%     [c2_prediction, c2_pr_positive] = BaseClassifierPredict(x(test_ind{1}, :), classifier_2);
%     y_test = y(test_ind{1}, :);
% 
%     % 置信度转成-1，1
%     c2_prediction(c2_prediction>=0) = 1; c2_prediction(c2_prediction<0) = -1;
% 
%     % ACCU_2
%     ACCU_2 = sum(c2_prediction == y_test(:, back_label));
%     disp("front_label = "+ front_label + ", back_label = " + back_label);
%     disp("单个的二分类情况下，ACCU_2 = " + ACCU_2);
% 
% 
%     % 三分类
%     y_train_2_3 = y_train(:,back_label);
%     y_train_2_3(indexes00) = 0;
%     y_train_2_3(indexes10) = 1;
%     y_train_2_3(indexes_1) = 2;
% 
%     classifier_3 = BaseClassifierTrain(y_train_2_3, x_train);
%     [c3_pr] = BaseClassifierPredict_new(x(test_ind{1},:), classifier_3);
% 
%     % 三分类结果转成二分类
%     c3_pr_new = zeros(size(y_test, 1), 3);
%     cr_pr_new(:, 1) = c3_pr(:, 1);
%     cr_pr_new(:, 2) = c3_pr(:, 2) + c3_pr(:, 3);
%     cr_pr_new(:, 3) = cr_pr_new(:, 1) - cr_pr_new(:, 2);
% 
%     cr_pr_new(cr_pr_new(:,3) >=0, 3) = 1;
%     cr_pr_new(cr_pr_new(:,3) <0, 3) = -1;
%     disp("置信度>0.5判定=1的情况下，ACCU_3 = " + sum(cr_pr_new(:,3) == y_test(:, back_label)));
% 
%     [temp1, temp2] = max((c3_pr'));
%     temp2 = temp2';
%     temp2(temp2 ~= 1) = -1;
%     disp("置信度在三者中值最大的情况下，ACCU_3 = " + sum(temp2 == y_test(:,back_label)));
%     disp("------------------------------------------------");
% 
%     cr_pr_a = zeros(size(y_test, 1), 50);
%     % 第一列存属于Lj的置信度
% 
% 
%     % 测试a取值不同，三分类效果如何
%     a = linspace(0, 1, 50);
%     accuracy_3 = zeros(1, 50);
%     %
%     for count = 1 : 50
%         cr_pr_a(cr_pr_new(:, 1) >= a(1, count), count) =  1;
%         cr_pr_a(cr_pr_new(:, 1) <  a(1, count), count) = -1;
%         accuracy_3(count) = sum(cr_pr_a(:, count) == y_test(:, back_label));
% 
%     end
% 
%     [max_value, max_index] = max(accuracy_3);
% 
%     temp = zeros(1, 50);
%     temp(:) = ACCU_2;
% 
% 
%     %             subplot(q,q-1, fig_index);
%     %             plot(a, accuracy_3, a, temp);
%     %
%     %             title("front label = " + front_label + ", back label = " + back_label);
%     %
%     %             fig_index = fig_index + 1;
% 
% 
% end
% 
