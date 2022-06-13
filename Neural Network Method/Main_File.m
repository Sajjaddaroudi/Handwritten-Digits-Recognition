clc
clear
load('Digtis_Data.mat');     %Loading gray scale data for each pixel of digit's image
%Network Features:
num_labels = 10; % 10 labels, from 1 to 10 (Output Layer' number of units)
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;    % 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
%Random Initialization:
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
%Using fmicg to train neural network:
options = optimset('MaxIter', 100);
lambda = 1;   %Regularization coefficent

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);       
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
%Predicting Output Layer
pred = predict(Theta1, Theta2, X);
%Accuracy of Training
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);