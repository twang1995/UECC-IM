function f1_macro_each_class = MacroF1_each_class(label,pred)

f1_arr = zeros(1, size(pred, 2));
for count = 1 : size(pred, 2)
    TP = sum((label(:, count)==1)&(pred(:, count)==1));
    FN = sum((label(:, count)==1)&(pred(:, count)==-1));
    FP = sum((label(:, count)==-1)&(pred(:, count)==1));

    epsilon = 1e-7;
    P = TP./(TP+FP+epsilon);
    R = TP./(TP+FN+epsilon);
    f1_arr(count) = mean(2*P.*R./(P+R+epsilon));

    TP = sum((label(:, count)==-1)&(pred(:, count)==-1));
    FN = sum((label(:, count)==-1)&(pred(:, count)==1));
    FP = sum((label(:, count)==1)&(pred(:, count)==-1));
    P = TP./(TP+FP+epsilon);
    R = TP./(TP+FN+epsilon);
    f1_arr(count) = f1_arr(count) + mean(2*P.*R./(P+R+epsilon));

    f1_arr(count) = f1_arr(count) / 2;


end

f1_macro_each_class = mean(f1_arr);


end



