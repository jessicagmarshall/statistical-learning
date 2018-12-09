% Jessica Marshall & Jason Katz
% ECE411: Recommendation Engine
% November 17, 2016

clc; clear all; close all
%% Format Data
filename = 'jester_data.csv';  
data = readtable(filename);               %row corresponds to user, column to joke rated (24984 total)
data = table2array(data); 

f = 2;         %# of factors

m = 1000; % #users
n = 100; % #items

rui = abs(data(1:m, :));     %select smaller portion of data

Y = abs(randn(n, f));
X = abs(randn(m, f));
alpha = 0.5;
C = 1 + alpha*rui;

lambda = 4;
numIter = 20;

est = zeros(1, numIter);    % for evaluating the estimate

for i = 1:numIter

    for j = 1:m
        X(j, :) = pinv(Y'*diag(C(j, :))*Y + lambda*eye(f)) * Y' * diag(C(j, :)) * C(j, :)';    %calculates xus
    end
    
    for k = 1:n
        Y(k, :) = (pinv(X'*diag(C(:, k))*X + lambda*eye(f)) * X' * diag(C(:, k)) * C(:, k))';     %calculates yis
    end
  
    rank = X*Y';

    est(i) = immse(rank, rui);
 
end

%groundtruth = ones(1, numIter)* rui;
xaxis = linspace(1, numIter, numIter);

%%
figure
plot(xaxis, est)
%plot(xaxis, groundtruth, xaxis, est)
title('convergence of NMF')
