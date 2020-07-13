function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

% m-dim vector representing the differences between our guess from hypothesis
% and the real value y (remember X is design matrix with 1st col of 1s)
difference = X*theta - y;

% take inner product is same as sum of squares
J = (1/(2*m)) * difference' * difference;
end



