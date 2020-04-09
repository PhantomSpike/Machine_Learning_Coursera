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
[m,n] = size(X);
K = size(Theta2,1); 
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

%Initialize the input activities to be equal to the numbers of pixels x
%training examples
a_1 = zeros(n+1,m);
%Add the a(1)_0 = bias term of ones for each training example
a_1(1,:) = ones; 
%Include the rest of the data
a_1(2:end,:) = X';
%Compute the weighted sum z_2 of the hidden layer
z_2 = Theta1*a_1;
%Apply a sigmoid activation function to every value
a_2 = sigmoid(z_2);
%Include bias terms a(2)_0 for every training example
a_2 = [ones(1,m);a_2];
%Compute the weighted sum z_3 of the output layer
z_3 = Theta2*a_2;
%Apply a sigmoid activation function to every value
a_3 = sigmoid(z_3);
%Explicitly state the hypothesis
h_theta = a_3;

%Convert y into a matrix of dimensions [training examples x number of
%outputs] i.e. 10x5000
y_i = zeros(K,m);
for ii = 1:numel(y)
    current_num = y(ii);
    y_i(current_num,ii) = 1;
end

%Compute the unregularized cost function summing over all training examples
%and all outputs
J = (-1/m)*(y_i(:)'*log(h_theta(:)) + (1-y_i(:)')*log(1 - h_theta(:)));
%Compute the regularization term for all weights
Theta1_reg = Theta1(:,2:end);
Theta2_reg = Theta2(:,2:end);
reg_term = (lambda/(2*m))*(Theta1_reg(:)'*Theta1_reg(:) + Theta2_reg(:)'*Theta2_reg(:));
%Compute the final regularized Cost function
J = J + reg_term;
%
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

Delta_2 = zeros(K,hidden_layer_size+1);
Delta_1 = zeros(hidden_layer_size,input_layer_size+1);
for ii = 1:m
    %Asign the ith training example x^(i) as an input a^(1)
    a_1 = X(ii,:);
    %Add the bias term a^(1)_0 
    a_1 = [1,a_1]';
    %Compute the weighted sum z_2 of the hidden layer
    z_2 = Theta1*a_1;
    %Apply a sigmoid activation function to every value
    a_2 = sigmoid(z_2);
    %Include bias terms a(2)_0 for every training example
    a_2 = [1;a_2];
    %Compute the weighted sum z_3 of the output layer
    z_3 = Theta2*a_2;
    %Apply a sigmoid activation function to every value
    a_3 = sigmoid(z_3);
    
    %Compute the error delta_3 for the output layer
    delta_3 = a_3 - y_i(:,ii);
    
    %Calculate the error delta_2 for the hidden layer
    delta_2 = Theta2(:,2:end)'*delta_3.*sigmoidGradient(z_2);
    
    %Accumulate the gradient for delta_3
    Delta_2 = Delta_2 + delta_3*a_2';
    
    %Accumulate the gradient for delta_2
    Delta_1 = Delta_1 + delta_2*a_1';
end

Theta2_grad = (1/m)*Delta_2;
Theta1_grad = (1/m)*Delta_1;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

reg_term_l2 = (lambda/m).*Theta1(:,2:end);
reg_term_l3 = (lambda/m).*Theta2(:,2:end);

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + reg_term_l2;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + reg_term_l3;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
