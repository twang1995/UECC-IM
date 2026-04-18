function model = BaseClassifierTrain(y,x)

%  BaseClassifierTrain function implements the training procedure of base
%  (binary) classifier
%
%  ========================================================================
%
%  Input:
%
%  - y: label vector of training examples, with length n. Each element
%       correspondings to an example's label and it's +1 or -1, standing 
%       for relevant label or irrelevant label.
%
%  - x:  feature matrix of training examples, with size n x d. Each row
%        corresponding to an example and each column corresponding to a
%        dimension. Due to LIBLINEAR's requirement, x must be in sparse 
%        format.
%
%  ========================================================================
%
%  Output:
%
%  - model:  trained model by base classifier. It will be used for 
%            predicting by BaseClassifierPredict function.

addpath('liblinear/matlab');

c_space = (10:40)/50;  % set of candidate parameter C in logistic regression
param_space = cellfun(@(c)sprintf('-s 6 -B 1 -c %.2f -q',c),...
    num2cell(c_space),'UniformOutput',false);  % generate parameter strings used in LIBLINEAR  % -s 6
param = SelectParameter(y,x,param_space);  % search optimal parameter through 5-fold cross validation

model = train(y,x,param);  % train a model by calling train functin implemented by LIBLINEAR

end

function param = SelectParameter(y,x,param_space)

param = param_space{1};
max_acc = train(y,x,[param,' -v 5']);
for i = 2:length(param_space)
    acc = train(y,x,[param_space{i},' -v 5']);
    if acc > max_acc
        param = param_space{i};
        max_acc = acc;
    end
end

end