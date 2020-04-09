function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
reg_param = [0.01,0.03,0.1,0.3,1,3,10,30];
num_params = numel(reg_param);
C_mat = [];
sigma_mat = [];

for C_number = 1:num_params
    C = reg_param(C_number);
    for sigma_number = 1:num_params
        sigma = reg_param(sigma_number);
        C_mat(C_number,sigma_number) = C;
        sigma_mat(C_number,sigma_number) = sigma;
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        pred_error(C_number,sigma_number) = mean(double(predictions ~= yval));
    end
end
[min_val,min_ix] = min(pred_error(:));
C = C_mat(min_ix);
sigma = sigma_mat(min_ix);


% =========================================================================

end
