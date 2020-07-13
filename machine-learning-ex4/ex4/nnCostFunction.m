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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Each row of X is a 400 dim vector
% representing one of our 5000 examples
% each example needs the bias unit added
% so we need to add a vector to first column of X
%   1 -  x^(1)  - .... 
%   1 -  x^(2)  - ....
%   .
%   .
%   1 -  x^(m)  - .....
% X has dim 5000 x 401 after bias unit
a1 = [ones(m, 1) X];

% Will use this to convert a given element of y to a num_labels-dim vector
% e.g. 3 might become 0 0 1 0 0 0 ....
labels_vector = (1: num_labels)';

% initialize cost function
J = 0;

% Note (AB)^T = B^TA^T
% and (AB^T)^T = BA^T
% so Theta1 * X^T = (X * Theta1^T)^T
% the difference is that on LHS the z2 results
% in a2 COLUMNS. Whereas on RHS we get a2 ROWS (just like the X had
% a row per training example).
% The following Theta2 * a2 = (a2^T * Theta2')^T  would lead
% to final h being rows too

% ======================== LOOP APPROACH========================================
#{
for i=1:m
  % each training example is the ith column
  z2 = Theta1 * (a1(i, :))';    % z2 is 25 x 1
  a2 = sigmoid(z2);   
  % add the bias unit
  a2 = [1; a2];  % a2 is 26 x 1
  z3 = Theta2 * a2;   % Theta2 is 10 x 26
  h = sigmoid(z3);

  % compare ith y result (e.g. y(i) =3 => (0 0 1 0 0 0 0 0 0 0)
  yvec = y(i) == labels_vector;
  % this is the sum over K via x^Tz=sum_i(x_i z_i)
  J = J + (1/m)*(-yvec'*log(h)-(1-yvec)'*log(1-h));
endfor
#}


% ========================MATRIX APPROACH ======================================

% Each column of the resulting matrix is one of the 5000 a1 vectors we had in
% the for loop approach. One 25-dim vector per training example
%  |      |    |  |    |
% a1_1  a1_2   .  .   a1_m
%  |      |    |  |    |
z2 = Theta1 * a1';  % 25 x 5000
a2 = sigmoid(z2); % 25 x 5000
% Adding the bias unit, just add a row of 1s to top of matrix
a2 = [ones(1, m); a2];  % 26 x 5000
% Each column of the resulting matrix is one of the 5000 a2 vectors we had in
% the for loop approach. One 10-dim vector per training example
%  |      |    |  |    |
% a3_1  a3_2   .  .   a3_m
%  |      |    |  |    |
h = sigmoid(Theta2 * a2); % 10 x 5000

% the y is a 5000 dimensional vector like (10 9 4 2 1....)
% we need to convert it to a 10x5000 matrix like
% 0  1 .
% 0  0 .
% 0  0 .
% .  . . 
% .  . .
% 1  0 .
y1 = zeros(num_labels, m);   % 10 x 5000
for k = 1:num_labels
    % this is kth row of y1 (sets it to 1 or 0 depending on if k is same as y)
    y1(k, :) = y == k;
end
% 
% y1(num_labels, :) = y == 0;
% one = ones(num_labels, m);
% we have a matrix of y where each col represents the mth training example
% value for y as a vector. and Each col of h is the hypothesis output for the 
% mth training example. Element-wise multiplication allows us to multipy every corresponding
% element of the 2 vectors e,g ym with hm then ...just like if we did it with transpose
% above. The second sum is over all m examples
J = sum(sum(-y1.*log(h)-(1-y1).*log(1-h)));
J = J / m;



% ========================REG ==================================================


% first col of theta corresponds to bias so can ignore
% (:, 2:end) gets everything but first column

reg1 = sum(sum(Theta1(:, 2:end) .* Theta1(:, 2:end)));
reg2 = sum(sum(Theta2(:, 2:end) .* Theta2(:, 2:end)));


J = J + (lambda/(2*m))*(reg1 + reg2);


% ========================BACKPROP==============================================


for i=1:m
  % ================== STEP 1 ==================================================
  % each training example is the ith column
  a1_i = (a1(i, :))';  % a1_i is 401x1
  z2 = Theta1 * a1_i;    % z2 is 25 x 1
  a2 = sigmoid(z2);   
  % add the bias unit
  a2 = [1; a2];  % a2 is 26 x 1
  z3 = Theta2 * a2;   % Theta2 is 10 x 26
  h = sigmoid(z3);

  % ==================== STEP 2 ================================================
  % compare ith y result (e.g. y(i) =3 => (0 0 1 0 0 0 0 0 0 0)
  yvec = y(i) == labels_vector;
  % How does our hypothesis compare with y
  delta3_vec = h - yvec;
  
  % ==================== STEP 3 ================================================

  % (:,2:end) drops the first col (the first col would always be the elements 
  % hitting the 1 in a2) leaving 10 x 25 matrix
  % after T is 25 x 10 matrix, and delta3_vec is 10 x 1 = > 25 x 1
  % product (same as a2). sigmoidGradient(z2) is a 25x1 just like z2
  delta2_vec = Theta2(:,2:end)' * delta3_vec .* sigmoidGradient(z2);
  
  % ==================== STEP 4 ================================================

  % delta2_vec is 25x1, yet a1_i' is 1x401  => 25x401 matrix (one for each ij)
  % d/dTheta_ij
  Theta1_grad = Theta1_grad + delta2_vec * a1_i';
  % delta3_vec is 10x1 and a2' is 1 x 26 => 10x26
  Theta2_grad = Theta2_grad + delta3_vec * a2';
endfor

% ==================== STEP 5 ================================================
% divide by m
Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

% ==================== REGULARIZE==============================================
% (:,2:end) means all but first column (which we leave intact) 
% (the first col of the theta corresponds to the elements that hit the bias
% unit...we don't want to regulize that)
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
