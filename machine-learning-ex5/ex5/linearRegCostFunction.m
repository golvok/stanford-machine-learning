function [J, grad] = linearRegCostFunction(X, y, weights, reg_importance)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, weights, reg_importance) computes the 
%   cost of using weights as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
data_size = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(weights));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of weights.
%
%               You should set J to the cost and grad to the gradient.
%



% =========================================================================

grad = grad(:);

end
