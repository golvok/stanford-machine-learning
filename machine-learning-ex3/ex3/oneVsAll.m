function [all_weights] = oneVsAll(X, y, num_labels, reg_importance)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_weights, where the i-th row of all_weights 
%corresponds to the classifier for label i
%   [all_weights] = ONEVSALL(X, y, num_labels, reg_importance) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_weights, where the i-th row of all_weights corresponds 
%   to the classifier for label i

% Some useful variables
data_size = size(X, 1);
num_features = size(X, 2);

% You need to return the following variables correctly 
all_weights = zeros(num_labels, num_features + 1);

% Add ones to the X data matrix
X = [ones(data_size, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter reg_importance.
%
% Hint: weights(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial weights
%     initial_weights = zeros(num_features + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal weights
%     % This function will return weights and the cost 
%     [weights] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), reg_importance)), ...
%                 initial_weights, options);
%

for label_num = 1:num_labels
	options = optimset('GradObj', 'on', 'MaxIter', 50);
	[new_weights] = fmincg (@(t)(lrCostFunction(t, X, (y == label_num), reg_importance)), transpose(all_weights(label_num,:)), options);
	all_weights(label_num,:) = transpose(new_weights);
end

% =========================================================================


end
