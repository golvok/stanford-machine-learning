function [J, grad] = costFunction(weights, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(weights, X, y) computes the cost of using weights as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of weights.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in weights
%
% Note: grad should have the same dimensions as weights
%

data_size = length(y);

hypo = sigmoid(X*weights);

J = (1/data_size)*sum(-y.*log(hypo) - (1 - y).*log(1-hypo));

grad = transpose((1/data_size)*sum((hypo - y).*X));

% =============================================================

end
