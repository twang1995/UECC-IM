function [ empEnt ] = expEnt( A )
%计算列向量的经验熵
empEnt=0;
list=unique(A);
l=length(list);
for i=1:l
    Length=length(find(A==list(i)));
    p=Length/length(A);
    empEnt=empEnt-p*log2(p);
end
 
 
end
 