function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

binary_labels = eye(num_labels); % Set binary representation of labels to binary_labels
binary_y = binary_labels(y, :); % Set binary representation of labels for each example to binary_y

% Begin - computing Feedforward
X = [ones(m, 1) X]; % Adding ones column
a1 = X;

z2 = a1 * Theta1';

a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2]; % Adding ones column
z3 = a2 * Theta2';
a3 = sigmoid(z3);
% End - computing Feedforwards

% Begin - Computing the cost function
sumarization = 0;
for i = 1:m
  for k = 1:num_labels
    sumarization = sumarization + (-binary_y(i, k ) * log(a3(i, k)) - (1 - binary_y(i, k)) * log( 1 - a3(i, k)));
  endfor
endfor

% Begin - Regularization
                      % Exclude the first columns (bias)
reg_theta1 = sum(sum((Theta1(:, 2:end) .^ 2)), 2); % Regularization for theta1
reg_theta2 = sum(sum((Theta2(:, 2:end) .^ 2)), 2); % Regularization for theta2

regularization = (lambda/( 2 * m)) * (reg_theta1 + reg_theta2);
% End - Regularization

J = (1/m) * sumarization + regularization;

% End - Computing the cost function

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
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

  d3 = a3 - binary_y;
  
  z2=[ones(m,1) z2];
  d2 = d3 * Theta2 .* sigmoidGradient(z2);
  
  d2 = d2(:, 2:end); % Removing bias
  
  D1 = a1' * d2;
  D2 = a2' * d3;
                               % Regularization part
  Theta1_grad = (1/m) * D1' + (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
  Theta2_grad = (1/m) * D2' + (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
