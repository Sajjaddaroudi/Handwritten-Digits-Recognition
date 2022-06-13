function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X=[ones(m,1) X];            %Adding x0=1 as first column to X matrix
z2=X* transpose(Theta1);
a_secondLayer= sigmoid(z2);  %activation of second layer
a_secondLayer= [ones(m,1) a_secondLayer];    %Adding a0=1 to a matrix 
z3=a_secondLayer* transpose(Theta2);
h_theta=sigmoid(z3);     %Output of the neural network
y_Matrix=zeros(length(y),num_labels);  
for i=1:length(y)
    y_Matrix(i,y(i))=1;
end
J = (-1/m) * sum((y_Matrix.*log(h_theta)+(1-y_Matrix)...
    .* log(1-h_theta)),'all') + (lambda/(2*m)) * (sum(Theta1(:,2:end).^2,'all')+...
    sum(Theta2(:,2:end).^2,'all'));
delta_3=h_theta - y_Matrix;
delta_2=(delta_3 * Theta2);
delta_2(:,2:end)=delta_2(:,2:end) .* sigmoidGradient(z2);
Delta_1=transpose(delta_2(:,2:end)) * X;
Delta_2=transpose(delta_3) * a_secondLayer;
Theta1_grad = (1/m) * Delta_1;
Theta2_grad = (1/m) * Delta_2;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * (Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * (Theta2(:,2:end));
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
