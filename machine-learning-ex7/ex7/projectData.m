function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

% take first K columns
U_red = U(:, 1:K);
% U_red is n x k, U_red' is k x n
% X is m x n
% alternatively Z=X*U_red;

% z^(i) = U^T * x^(i);
% z^(i)_p = (U^T)_pq * x^(i)_q;
% But if we define Z in same way as X (i.e. each row is one of m examples, but
% now in k-dimensional space, not n-dimensional, so Z is mxk), then z^(i)_p 
% equals Z_ip. Also remembering that x^(i)_q = X^T_qi (qth component of ith example
% is ith col of X^T in qth row - it's X^T rather than X because x is col vec)
% Z_ip = U^T_pq * X^T_qi;
% Z_ip = (U^T*X^T)_pi =
% Z = (U^T*X^T)^T;
% Z = X*U;
Z = (U_red'* X')';  

% =============================================================

end
