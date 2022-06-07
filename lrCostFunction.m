function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
theta2=theta.^2;  %theta to the power of 2
J =(-1/m) * sum ( y.* log(sigmoid(X*theta)) + (1-y).* log(1-sigmoid(X*theta)))...
    + (lambda/(2*m))* (sum (theta2(2:end,1)));     %Regularized cost function
grad=(1/m) * transpose(X) * (sigmoid(X*theta)-y);   %Gradient without Regulariztion
grad(2:end)=grad(2:end)+(lambda/m)* theta(2:end);   %Gradient with Regularization

end
