function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, reg_importance)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, reg_importance) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
data_size = size(X, 1);

% You need to return these values correctly
error_train = zeros(data_size, 1);
error_val   = zeros(data_size, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the reg_importance argument set to 0. 
%       Do note that you will still need to use reg_importance when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:data_size
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------

for i = 1:data_size
	X_subset = X(1:i, :);
	y_subset = y(1:i);

	weights = trainLinearReg(X_subset, y_subset, reg_importance);
	[ error_train(i), dummy ] = linearRegCostFunction(X_subset, y_subset, weights, 0);
	[ error_val(i), dummy ] = linearRegCostFunction(Xval, yval, weights, 0);
end

% -------------------------------------------------------------

% =========================================================================

end
