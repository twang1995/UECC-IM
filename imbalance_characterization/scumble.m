function [variable_scumble] = scumble(y)
% 并发级别度量：评估非常频繁和罕见的标签之间的并发性
% ( assess the concurrence among very frequent and rare labels )。
% 值小，表示MLD在不平衡标签之间没有太多共现性；值越大，表示越有共现性。

% 想法：探索两个标签间的并发性
% 似乎实现上是相同的，只需插入时放入完整的Q个标签，还是2个标签的问题。
% scumble_paper中存在问题，即累乘总是为0，对此进行了改进
% scumble.m中，只统计属于的标签，其他标签忽略不统计。

[m, q] = size(y);

variable_scumble = 0;

for i = 1 : m
    IRLbl_i_mean = 0;
    for lambda = 1 : q
        if y(i, lambda) == 1
            IRLbl_i_mean = IRLbl_i_mean + IRLbl(y, lambda);
        end
    end
    IRLbl_i_mean = IRLbl_i_mean / sum( y(i, :) == 1 );

   if ~isnan(IRLbl_i_mean)
        IRLbl_i_lambda_product = 1;
        for lambda = 1 : q
            if y(i, lambda) == 1
                IRLbl_i_lambda_product = IRLbl_i_lambda_product + IRLbl(y, lambda);
            else
%                             IRLbl_i_lambda_product = IRLbl_i_lambda_product * 0;
            end
        end
        labels_belong_number = sum( y(i, :) == 1 );
        labels_belong_number = q;
        variable_scumble = variable_scumble + (1 -  (1 / IRLbl_i_mean) * power(IRLbl_i_lambda_product, 1/ labels_belong_number) );
    end
end

variable_scumble = variable_scumble / m;

end
