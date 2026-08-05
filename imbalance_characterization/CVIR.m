function [variable_CVIR] = CVIR(y)
%CVIR IRLbl变异系数 CVIR ： CVIR衡量IRLbl的变异程度，即所有标签之间不平衡程度的相似度。
 %它表明标签是否经历类似程度的不平衡，或者它们之间是否存在很大差异。 
 % CVIR值越大，差异越大。

q = size(y, 2);

IRLbl_sigma = 0;
for lambda = 1 : q
    IRLbl_sigma = IRLbl_sigma + ((IRLbl(y, lambda) - meanIR(y)) ^ 2 / (q-1));
end

variable_CVIR = IRLbl_sigma / meanIR(y);

end

