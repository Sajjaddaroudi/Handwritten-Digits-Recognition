function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
X = [ones(m, 1) X];
Prob_matrix= X* transpose(all_theta);   % Matrix which shows the probability of 
                                           % each test be equal with one of
                                           % the possible numbers
[~,p] = max(Prob_matrix,[],2);          %Assigning index with maximum probability in each
                                            % row as predicted number for
                                            % that row's input data

end
