function [prediction,pr_positive] = BaseClassifierPredict(x,model)
% function [pr] = BaseClassifierPredict(x,model)


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

%[predicted_label, accuracy, prob_estimates] = predict(nan(size(x,1),1),x,model,'-b 1 -q');

%size(prob_estimates)
%prediction = predicted_label;

prediction = pr(:,1)- pr(:,2);
pr_positive = pr(:,1);


%prediction = predicted_label;
end
