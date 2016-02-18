function [cost, grad] = lrCostFunction(weights, X, y, reg_importance)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   cost = LRCOSTFUNCTION(weights, X, y, reg_importance) computes the cost of using
%   weights as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

data_size = length(y);

hypo = sigmoid(X*weights);

cost = (1/data_size)*sum(-y.*log(hypo) - (1 - y).*log(1-hypo)) + reg_importance/(2*data_size)*sum(weights([2:length(weights)]).^2);

grad = transpose((1/data_size)*sum((hypo - y).*X)) + [ 0; reg_importance/data_size*weights([2:length(weights)]) ];

end
