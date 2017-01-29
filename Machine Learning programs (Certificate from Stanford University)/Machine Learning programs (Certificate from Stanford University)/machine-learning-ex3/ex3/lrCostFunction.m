function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% Unvectorized cost function 
theta_from2 = [0; theta(2:end)]; % Here I set theta0 to 0 and practically I care for the other thetas,e.g. theta1,theta2 etc.
J = -(sum(y'*log(sigmoid(X*theta))+(1-y')*log(1-sigmoid(X*theta)))/m); 
J = J + (lambda/(2*m))*(theta_from2'*theta_from2); % The multiplication of the theta_from2 matrix by its inversed matrix is practically the square operation
% Vectorized cost function
% theta_from2 = [0; theta(2:end)];
% J = X*theta;
% J = J + (lambda/(2*m))*(theta_from2'*theta_from2);

grad = X'*(sigmoid(X*theta)-y)/m + (lambda/m)*theta_from2;
grad = grad(:);

end
