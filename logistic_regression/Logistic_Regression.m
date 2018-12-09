% Jessica Marshall
% ECE411: Logistic Regression 
% October 2, 2016

% Binary Classification of Wine Using Stochastic Gradient Descent

clc; clear all; close all;

%% Read in Data

filename = 'Wine_Classification.xlsx';

X = xlsread(filename, 'Q2:AD132');
y = xlsread(filename, 'A2:A132');          % {0,1} corresponding to wine category
[zzz, ftrnames, yyy] = xlsread(filename, 'Q1:AD1');       % feature names

%% Initial Logistic Regression

numObs = length(y);       % # observations (131)
n = size(X);
numFeat = n(2);           % # of features including bias (14)

init_theta = .01 * ones(numFeat, 1);            % initial guess for theta, excluding theta0

hx = sigmoid(X(1,:)*init_theta);        % compute initial hx using first observation initial theta

%classify observations

pre_probs = zeros(numObs, 1);
pre_label = zeros(numObs, 1);

for i = 1:numObs
    pre_probs(i) = sigmoid(X(i, :)*init_theta);
    
    if pre_probs(i) > .5
        pre_label(i) = 1;
        
    end
end

theta = init_theta;
pre_numWrong = sum(abs(y-pre_label));
pre_accuracy = (numObs - pre_numWrong) / numObs

%SGD

numIter = 10000;
alpha = .05;

log_like = zeros(numIter, 1);       % to plot log likelihood

for k = 1:numIter
    
    i = randi([1 numObs],1);        % randomly choose observation
    
    for j = 1:numFeat
        theta(j) = theta(j) + alpha*(y(i) - sigmoid(X(i,:)*theta))*X(i, j);        % update theta
    end
    
    post_probs = zeros(numObs, 1);
    post_label = zeros(numObs, 1);

    for i = 1:numObs
        post_probs(i) = sigmoid(X(i, :)*theta);      % recalculate hx's of each observation using new theta
    
        if post_probs(i) > .5
            post_label(i) = 1;
        
        end
    end
    
    log_like(k) = sum(y'*log(post_probs) + (1-y')*log(1-post_probs));
    post_numWrong = sum(abs(y-post_label));             %calculate accuracy
    post_accuracy = (numObs - post_numWrong) / numObs;

end

post_accuracy

itervec = linspace(1, numIter, numIter);
plot(itervec, log_like)
title('Convergence of SGD', 'Fontsize', 20,'FontWeight','normal','FontName','Cambria')

xlabel('Number of Iterations','FontSize',18, 'FontName', 'Cambria')

ylabel({'Log Likelihood'}, 'FontSize',18, 'FontName', 'Cambria')