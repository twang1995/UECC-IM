function [Ecents] = Ecents( A )
%计算信息增益
size_A=size(A);
list_A=A(:,size_A(2));
empEnt=expEnt(list_A);
Ecents=zeros((size_A(2)-1),1);
for i=1:size_A(2)-1
    data=A(:,i);
    Ecents(i)=Ecent(data,list_A,empEnt);
end
    
 
 
end
 