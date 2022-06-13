clc
clear
load('Digtis_Data.mat');     %Loading gray scale data for each pixel of digit's image 
num_labels = 10; % 10 labels, from 1 to 10 
lambda = 0.1;       %Regularization coefficent
[all_theta] = oneVsAll(X, y, num_labels, lambda);   %Obtaining each class's hypothesis
pred = predictOneVsAll(all_theta, X);        %Predicting each image's corresponding digit
%Calculating the accuracy of our prediction
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);