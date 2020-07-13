function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%1/(1+exp(-z))
% z^(i)=theta_0*x_0^(i)+theta_1+x_1^(i)+.......+theta_n*x_n^(i)
% z^(i) = (Xtheta)^(i)
guess_input = X*theta;
h = sigmoid(guess_input);

J = (1/m)*(-y'*log(h)-(1-y)'* log(1-h));

% how much our guess h differs from y for each of the m examples
difference = h-y;

% X = (<---------x^0 ----------->)
%      (<---------x^1 ------------>)
% .
% .
%      (<---------x^m ------------>)
% so the difference vector multiple down the first column
% for example to get first component of grad....across the m examples
% but always zeroth component 
grad = (1/m)*(difference)'*X;



% =============================================================

end
