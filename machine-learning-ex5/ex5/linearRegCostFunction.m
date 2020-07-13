function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Remember each row of X is one of the m training examples
% ex5 already adds the ones bias to give a 12x2 matrix, theta is 2x1 => 12x1
h = X*theta;
diff = h - y;

% ignore the bias theta when regularizing (theta0 is lec notes, theta1 here)
J = (diff' * diff) + (lambda * theta(2:end)' * theta(2:end));
J = J/(2*m);

% =========================================================================

grad = grad(:);

% unreg looks like
% 1/m (diff_i * X_ij)   <- because each row is ith training example =>
% i is the column index. So 1/m X^T_ji * diff_i = 1/m X^T diff
grad = (X' * diff)/m;

% add reg for all but bias term (note it's important this generalized
% to more than 2 dims to pass. So grad(2) = grad(2) + .. doesnt work
grad = grad + (lambda/m)* [0; theta(2:end)];

end
