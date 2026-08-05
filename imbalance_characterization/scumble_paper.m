% 并发级别度量：评估非常频繁和罕见的标签之间的并发性
% ( assess the concurrence among very frequent and rare labels )。
% 值小，表示MLD在不平衡标签之间没有太多共现性；值越大，表示越有共现性。

% 想法：探索两个标签间的并发性
% 似乎实现上是相同的，只需插入时放入完整的Q个标签，还是2个标签的问题。

function [variable_scumble] = scumble_paper(y)
% 形参y的size是[样本数量 * 标签数量]

% 样本数量m，标签数量
[m, q] = size(y);

variable_scumble = 0;

for i = 1 : m
    IRLbl_i_mean = 0;
    for lambda = 1 : q
        if y(i, lambda) == 1
            IRLbl_i_mean = IRLbl_i_mean + IRLbl(y, lambda);
        end
    end
    IRLbl_i_mean = mean(IRLbl_i_mean);


    IRLbl_i_lambda_product = 1;
    for lambda = 1 : q
        if y(i, lambda) == 1
            IRLbl_i_lambda_product = IRLbl_i_lambda_product * IRLbl(y, lambda);
        else
%             IRLbl_i_lambda_product = IRLbl_i_lambda_product * 0;
        end
    end



    variable_scumble = variable_scumble + (1 -  (1 / IRLbl_i_mean) * power(IRLbl_i_lambda_product, 1/q ) );
end

variable_scumble = variable_scumble / m;


end



