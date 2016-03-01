function [reg_importance_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select reg_importance
%   [reg_importance_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of reg_importance. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of reg_importance (you should not change this)
reg_importance_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(reg_importance_vec), 1);
error_val = zeros(length(reg_importance_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector reg_importance_vec contains the different reg_importance parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               reg_importance = reg_importance_vec(i)
%
% Note: You can loop over reg_importance_vec with the following:
%
%       for i = 1:length(reg_importance_vec)
%           reg_importance = reg_importance_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter reg_importance
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%

% =========================================================================

end
