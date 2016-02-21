function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, reg_importance)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, reg_importance) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Weights1 and Weights2, the weight matrices
% for our 2 layer neural network
Weights1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Weights2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
data_size = size(X, 1);
num_features = size(X, 2);
         
% You need to return the following variables correctly 
J = 0;
Weights1_grad = zeros(size(Weights1));
Weights2_grad = zeros(size(Weights2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Weights1_grad and Weights2_grad. You should return the partial derivatives of
%         the cost function with respect to Weights1 and Weights2 in Weights1_grad and
%         Weights2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Weights1_grad
%               and Weights2_grad from Part 2.
%
% -------------------------------------------------------------

X = [ ones(data_size,1) , X ];

inputs_l2 = X*transpose(Weights1);
output_l2 = [ ones(data_size,1), sigmoid(inputs_l2) ];
inputs_l3 = output_l2*transpose(Weights2);
output = sigmoid(inputs_l3);

for i = 1:data_size
	onehot_y = zeros(num_labels,1);
	onehot_y(y(i)) = 1;

	J += sum(-onehot_y.*log(transpose(output(i,:)))-(1-onehot_y).*log(1-transpose(output(i,:))));
end

J = J./data_size;

% =========================================================================

% Unroll gradients
grad = [Weights1_grad(:) ; Weights2_grad(:)];


end
