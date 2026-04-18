function [gain_ratio] = Gain_ratio( A,B)

gain_ratio = 0;
emEnt = expEnt(A);
gain_ratio = Ecent( A,B,emEnt);


end