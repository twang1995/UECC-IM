function [scumble_front_back] = Copy_of_scumble_between_labels(y, front_label, back_label)
%   scumble.m中若对两个标签进行测量，是对称的。
%   但实际效果中，1->2和2->1，三分类的效果是不一样的。
%   因此，scumble_between_labels.m解决该问题，变成不对称的，反应出前后的效果。

%   【不同】scumble中如果只放入L1
%   L2，则IRLbl_l1和IRLbl_l2与放入y_train时的IRLbl_l1和IRLbl_l2不同。
%   构思：。。。。

[m, q] = size(y);



%   计算每个标签的IRLbl，存储到1*q的IRLbl_each_labels
IRLbl_each_labels = zeros(1, q);
for lambda = 1 : q
    IRLbl_each_labels(1, lambda) = IRLbl(y, lambda);
end

%   scumble_matrix存储标签对的并发情况度量，其中每个标签的IRLbl是基于y的，而不是仅基于两个标签计算出来的。
scumble_matrix = zeros(q, q);

scumble_front_back = 0;
for i = 1 : m

    IRLbl_i_mean = 0;
    count_front = 0;
    count_back = 0;
    IRLbl_i_lambda_product = 1;
    if y(i, front_label) == 1
        IRLbl_i_mean = IRLbl_i_mean + IRLbl_each_labels(front_label);
        count_front = 1;
    end

    if y(i, back_label) == 1
        IRLbl_i_mean = IRLbl_i_mean + IRLbl_each_labels(back_label);
        count_back = 1;
    end

%     if count_front == 0 && count_back == 1
%         IRLbl_i_lambda_product = IRLbl_i_lambda_product * IRLbl_each_labels(back_label);
%     end

    if count_front == 1 && count_back == 1
        IRLbl_i_lambda_product = IRLbl_i_lambda_product * IRLbl_each_labels(front_label) * IRLbl_each_labels(back_label);
    end

    if count_front ~= 0 && count_back ~= 0
        temp = (1 - (1 / IRLbl_i_mean) * power(IRLbl_i_lambda_product, 1/(2)));
        scumble_front_back = scumble_front_back + temp;

    end

end

scumble_front_back = scumble_front_back / m;

end