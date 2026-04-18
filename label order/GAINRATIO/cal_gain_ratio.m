function [ gain_ratio ] = cal_gain_ratio(D,a)


% Gain_ratio(D, a) = Gain(D, a) / IV(a)
% 信息增益 / 熵
% 注意：变量的顺序
gain_ratio = Ecents([D,a]) / expEnt(a);

