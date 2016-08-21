function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%disp('size of theta = ');
%size(theta);
%disp('size of X = ');
%size(X);
%disp('size of y = ');
%size(y);
h_theta = X*theta;       

%disp('size of h_theta = ');
%size(h_theta);

%J for Linear Regression + Regularization
J = sum((h_theta - y).^2) / m/ 2 + lambda/2/m*sum(theta(2:end,1).^2);
%disp('size of J = ');
%size(J);

%Gradient for Linear Regression + Regularization
grad(1,1) = (1/m)*sum((h_theta-y).*X(:,1)); 
grad(2:end,1)=((1/m)*((h_theta-y)'*X(:,2:end)))'+(lambda/m)*theta(2:end);
%disp('size of grad = ');
%size(grad);





% =========================================================================

grad = grad(:);

end
