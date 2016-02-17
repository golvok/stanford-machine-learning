function [J, grad] = costFunctionReg(weights, X, y, reg_importance)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(weights, X, y, reg_importance) computes the cost of using
%   weights as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of weights.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in weights

data_size = length(y);

[ J, grad ] = costFunction(weights, X , y);

J += reg_importance/(2*data_size)*sum(weights([2:size(weights)]).^2)

grad += [ 0; reg_importance/data_size*weights([2:size(weights)]) ];

% =============================================================

end
