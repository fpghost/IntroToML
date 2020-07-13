function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

% Randomly reorder the indices of examples
% e.g. if we had [1, 2; 3, 4] so size(, 1) is 2 then randperm
% gives 1, 2 or 2, 1...randperm(3) 3 2 1 or 2 1 3 etc
randidx = randperm(size(X, 1));
% Take the first K examples as centroids
% cherry pick out these K rows
centroids = X(randidx(1:K), :);






% =============================================================

end

