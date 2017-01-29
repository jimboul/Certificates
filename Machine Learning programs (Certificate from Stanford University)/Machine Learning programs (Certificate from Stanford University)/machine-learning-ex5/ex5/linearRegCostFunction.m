function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% Cost function
h = X*theta;
theta_from2 = [0; theta(2:end)];
J = sum((h - y).^2)/(2*m);
reg = (lambda/(2*m))*(theta_from2'*theta_from2);
J = J + reg;

% Gradient descent
grad = X'*(h - y)/m + (lambda/m)*theta_from2;
grad = grad(:);

end
