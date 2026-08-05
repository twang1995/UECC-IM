function [variable_meanIR] = meanIR(y)
%Cal_meanIR 平均不平衡率
%   1/q * sum_{lambda \belong L} IRLbl(lambda)

q = size(y, 2);
IRLbl_all = zeros(1, q);

for lambda = 1 : q
    IRLbl_all(1, lambda) = IRLbl(y, lambda);
end

variable_meanIR = 1 / q * sum(IRLbl_all);


end

