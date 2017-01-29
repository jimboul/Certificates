function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

C = 1;
sigma = 0.3;
results = eye(64,3);
error_counter = 0;
for test_C = [0.01 0.03 0.1 0.3 1 3 10 30]
    for test_sigma = [0.01 0.03 0.1 0.3 1 3 10 30]
        error_counter = error_counter + 1;
        model = svmTrain(X, y, test_C, @(x1, x2) gaussianKernel(x1, x2, test_sigma));
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));
        results(error_counter,:) = [test_C; test_sigma; prediction_error];
    end
end
sorted_results = sortrows(results,3);
C = sorted_results(1,1);
sigma = sorted_results(1,2);

end
