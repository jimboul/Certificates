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
k = 0;

%J = -(1/m).*symsum(log(1./(1+exp(-(theta*X))))*y+log(1-(1./(1+exp(-(theta*X)))))*(1-y),k,1,m);
%grad = (1/m)*symsum((1./(1+exp(-(X*theta)))-y)*X,k,1,m);

J = sum(-y' * log(1./(1+exp(-(X*theta)))) - (1 - y')*log(1 -(1./(1+exp(-(X*theta)))))) / m;
grad = X' * ((1./(1+exp(-(X*theta)))) - y) / m;


end
