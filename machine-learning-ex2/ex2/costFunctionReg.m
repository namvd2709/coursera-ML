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

sigmoid = sigmoid(X*theta)
for i=1:m,
  J += 1/m*(-y(i,1)*log(sigmoid(i,1)) - (1-y(i,1))*log(1-sigmoid(i,1)));
end;

for j=2:rows(theta),
  J += lambda/(2*m)*(theta(j,1))**2;
end;

for i=1:m,
  grad(1,1) += 1/m*(sigmoid(i,1)-y(i,1))*X(i,1);
end;

for i=2:rows(theta),
  for j=1:m,
    grad(i,1) += 1/m*(sigmoid(j,1) - y(j,1))*X(j,i);
  end;
end;

for i=2:rows(theta),
  grad(i,1) += lambda/m*theta(i,1);
end;




% =============================================================

end
