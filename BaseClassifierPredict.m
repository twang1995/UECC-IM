% function prediction = BaseClassifierPredict(x,model)
function [prediction,pr] = BaseClassifierPredict(x,model)


%  BaseClassifierPredict function implements the predicting procedure of 
%  base (binary) classifier
%
%  ========================================================================
%
%  Input:
%
%  - x:  feature matrix of testing examples. Each row corresponding to an 
%        example and each column corresponding to a dimension. Due to 
%        LIBLINEAR's requirement, x must be in sparse format.
%
%  - model:  trained model by BaseClassifierTrain function.
%
%  ========================================================================
%
%  Output:
%
%  - prediction:  predictive values of testing examples. Each element 
%                 corresponding to an example and its value is in [-1,1].

addpath('liblinear/matlab');

[~,~,pr] = predict(nan(size(x,1),1),x,model,'-b 1 -q');

if size(pr, 2) == 1
    pr = [pr, pr];
    if model.Label == 1
        pr(:, 2) = 0;
    end
    if model.Label == -1
        pr(:, 1) = 0;
    end
end


    prediction = pr(:,1) - pr(:,2);
    prediction(prediction > 0) = 1;
    prediction(prediction < 0) = 0;


end
