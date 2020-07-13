function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% X = num_movies x num_features
% X = (-------------x^(1)-------------)
%     (-------------x^(2)-------------)
%     (-------------x^(3)-------------)
%     ...
%     (-------------x^(num_movies)-----)
% Theta = num_users x num_features
% Theta = (----------theta^(1)----------)
%         ...
%         (----------theta^(num_users) ----)

% the predicted recommenation for jth user of movie i
% (theta^(j))^T*x^(i) = (X*Theta')_ij = (Theta*X')_ji

% This is prediction, and is  num_movies x num_users
% The ij element is rating of movie i by user j
P = X * Theta';
% R is num_movies x num_users and the ij element is if rating of movie i by user j
% element-wise multiply (note if used Theta*X' would need transpose of R)
% since R would be zero if not rated, this sets all ij 0 of P when R_ij=0
% and that will have the same effect as limiting our sum over rows and columns
% to only those for which R != 0
P = P.*R;


DIFF = P - Y;


J = sum(sum(DIFF.*DIFF))/2;

reg_theta = (lambda/2) * sum(sum(Theta.^2));
reg_x = (lambda/2) * sum(sum(X.^2));

J = J + reg_theta + reg_x;

% Gradients

% DIFF is num_movies x num_users
% theta^(j)_k, means how user  j rated movie k. We are fixing k and will work
% down j, meaning we take the kth col
% see notebook
X_grad = DIFF * Theta;

X_grad = X_grad + lambda*X;

% x^(i)_k, means the kth component of ith vector, 
% See notebook
Theta_grad = DIFF' * X;

Theta_grad = Theta_grad + lambda*Theta;




% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
