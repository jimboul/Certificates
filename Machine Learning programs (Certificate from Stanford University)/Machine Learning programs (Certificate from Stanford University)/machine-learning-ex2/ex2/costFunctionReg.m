function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


[J, grad] = costFunction(theta, X, y);
theta_from2 = [0; theta(2:end)];
J = J + ((lambda / (2*m)) * (theta_from2' * theta_from2));
grad = grad + ((lambda / m) * theta_from2);

end
