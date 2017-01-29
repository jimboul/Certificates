function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be an "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%Y = eye(K)(y,:);
%Y = subsref(eye(num_labels),struct('type','()','subs',{{y,:}}));
% paren = @(x, varargin) x(varargin{:});
% curly = @(x, varargin) x{varargin{:}};
% Y = paren(eye(num_labels),y,:);
% subindex = @(A,r,c) A(r,c);
% Y = subindex(eye(num_labels),y,:); 
Y = eye(num_labels);
Y = Y(y,:);

a1 = [ones(m,1),X];
z2 = Theta1*a1';
a2 = sigmoid(z2);
a2 = [ones(1,size(a2,2)); a2];
z3 = Theta2*a2;
h = sigmoid(z3);
cost = Y'.*log(h) + (1-Y').*log(1-h); %It may be needed to write it as follows:
                                      % cost = Y.*log(h)' + (1-Y).*log(1-h)';
                                      % (if wrong the already written
                                      % code)!!!
J = -(1/m)*sum(cost(:)); % Unregularized cost function
Theta1_from2 = Theta1(:,2:end);
Theta2_from2 = Theta2(:,2:end);
reg = (lambda/(2*m))*(sumsqr(Theta1_from2(:)) + sumsqr(Theta2_from2(:)));
J = J + reg; % Regularized cost function

% Backpropagation Algorithm
Delta_2 = 0;
Delta_1 = 0;
for t = 1:m
    a1 = [1; X(t,:)']; % It may be needed to use inverse X matrix,that is X(t,:)'
    z2 = Theta1*a1; % It may be needed to use simple a1 (not inversed as here!!!)
    a2 = [1; sigmoid(z2)];
    z3 = Theta2*a2;
    a3 = sigmoid(z3);
    yt = Y(t,:)';
    d3 = a3 - yt;
    d2 = (Theta2_from2'*d3).*sigmoidGradient(z2);
    Delta_2 = Delta_2 + d3*a2';
    Delta_1 = Delta_1 + d2*a1';
end
Theta1_grad = (1/m)*Delta_1;
Theta2_grad = (1/m)*Delta_2;
    
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1_from2;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2_from2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
