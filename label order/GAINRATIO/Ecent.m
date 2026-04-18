function [ ecEnt ] = Ecent( A,B,emEnt)
%计算经验条件熵
%第三个参数emEnt是A的熵
ecEnt=emEnt;
Lengh_A=length(A);
list=unique(A);
Length=length(list);
for i=1:Length
    loc=find(A==list(i));
    L=length(loc);
    Save=zeros(L,1);
    for j=1:L
        Save(j)=B(loc(j));
    end
   ecEnt=ecEnt-expEnt(Save)*L/Lengh_A;
end
end
 