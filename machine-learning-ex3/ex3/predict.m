function predictions = predict(Weights1, Weights2, X)
%PREDICT Predict the label of an input given a trained neural network
%   predictions = PREDICT(Weights1, Weights2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Weights1, Weights2)

% Useful values
data_size = size(X, 1);
num_labels = size(Weights2, 1);

% You need to return the following variables correctly 
predictions = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set prediction to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%



% =========================================================================


end
