function macro_f1 = MacroF1_majority(label,pred)
% 由于每个标签下，少数类的值可能不一样，所以使用时只针对单标签衡量，而后再进行平均

if sum(label== -1) >= sum(label == 1)
    min_value_back_label = 1;
    %             min_samples_num = sum(label(:, back_label) == 1);
else
    min_value_back_label = -1;
    %             min_samples_num = sum(label(:, back_label) == -1);
end

TP = sum((label ~= min_value_back_label)&(pred ~= min_value_back_label));
FN = sum((label ~= min_value_back_label)&(pred == min_value_back_label));
FP = sum((label == min_value_back_label)&(pred ~= min_value_back_label));

epsilon = 1e-7;
P = TP./(TP+FP+epsilon);
R = TP./(TP+FN+epsilon);

macro_f1 = mean(2*P.*R./(P+R+epsilon));
end



