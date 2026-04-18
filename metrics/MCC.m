function mcc_value = MCC(label,pred)
%MCC 计算MCC

TP = sum((label==1)&(pred==1));
FN = sum((label==1)&(pred~=1));
FP = sum((label~=1)&(pred==1));
TN = sum((label~=1)&(pred~=1));

epsilon = 1e-7;

mcc_value = (TP .* TN - FP .* FN) ./ ( sqrt( (TP + FP) .* (TP +FN) .* (TN + FP) .* (TN + FN) ) + epsilon );


end



