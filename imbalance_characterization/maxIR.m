function IRLbl_max = maxIR(y)
%   maxIR q个标签中IRLbl的最大值
%   minIR不用算，因为minIR一定等于1

q = size(y, 2);
IRLbl_max = -1;

for lambda = 1 : q
    IRLbl_now = IRLbl(y, lambda);
    if IRLbl_max < IRLbl_now
        IRLbl_max = IRLbl_now;
    end

end



end


