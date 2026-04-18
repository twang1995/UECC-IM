function IR = cal_IR(y)
%CAL_IR 计算当前标签的IR

    majority_number = max(sum(y == 1), sum(y ~= 1));
    minority_number = min(sum(y == 1), sum(y ~= 1));
    IR = majority_number / minority_number;

end

