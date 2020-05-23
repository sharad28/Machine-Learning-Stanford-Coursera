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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

g_z=sigmoid(X*theta); % m x1 therefore (g_z-y) is also m x 1
J=(1/m)*(-y'*log(g_z)-(1-y')*(log(1-g_z)))+(lambda/(2*m)) * (theta(2:end)'*theta(2:end));

grad(1)=(1/m)*(X(:,1)'*(g_z-y));

grad(2:end)=(1/m)*(X(:,2:end)'*(g_z-y))+lambda/m*theta(2:end);





% =============================================================

end
