%Jessica Marshall & Jason Katz
%ECE411 Linear Regression Exercise
%September 25, 2016

clc; clear all; close all;

fprintf('We use Happiness Index raw data to determine which features have the most effect on the freedom of an\nindividual to make life choices.\n')

%% Table 3.1: Correlation Matrix of Components of Happiness Score

filename = 'Linear_Regression.xlsx';

X = xlsread(filename, 'E2:O907');
[zzz, ftrnames, yyy] = xlsread(filename, 'E1:O1');       %feature names
R = corrcoef(X);                    % correlation matrix

%format table1
titlerow1 = cell(ftrnames);           % make title row
rowlabels1 = cell(ftrnames');      % make row labels

fprintf('The table below corresponds to Table 3.1 of the prostate example:');
fprintf('\n')
fprintf('\n')

table1 = array2table(R, 'VariableNames', titlerow1, 'RowNames', rowlabels1);  %output table 3.1
disp(table1)
%% Table 3.2: Linear Model Fit to Happiness Index Data

y = xlsread(filename, 'D2:D907');
beta_hat = regress(y,X);        % betas hats for each feature
y_hat = X*beta_hat;

v = inv(X'*X);
sigma_hat = sqrt(1 / (size(X, 1) - size(X, 2) - 1)) * sum((y - y_hat).^2);
std_err = sigma_hat * sqrt(diag(v));        
z_score = beta_hat ./ std_err;      % z score of each feature

%format table2
tablearray = [beta_hat std_err z_score];

titlerow2 = {'Coefficient'; 'Standard_Error'; 'Z_score'};
rowlabels2 = cell(ftrnames');

fprintf('The table below corresponds to Table 3.2 of the prostate example:');
fprintf('\n')
fprintf('\n')

table2 = array2table(tablearray, 'VariableNames', titlerow2, 'RowNames', rowlabels2);  %output table 3.1
disp(table2)

%% Table 3.3: Estimated Coefficients and Test Error Results for Different Subset and Shrinkage Methods

%best subset - choose 3 features using forward search
min_sq_err_i = 1000;
ft_index_i = 0;

for i = 1:11                            % find first feature
    
    beta_hat_i = regress(y,X(:,i));
    y_hat_i = X(:,i) * beta_hat_i;
    sq_err_i = sum((y_hat_i - y).^2);
    
    if sq_err_i < min_sq_err_i
        ft_index_i = i;               % X index of best feature
        min_sq_err_i = sq_err_i;
    end
end

min_sq_err_j = 1000;
ft_index_j = 0;
    
for j = 1:11                            % find second feature
    if j == ft_index_i
        continue
    end

    Xj = [X(:, ft_index_i) X(:, j)];          % concatenate 2 features together

    beta_hat_j = regress(y, Xj);     % perform regression on 2 features
    y_hat_j = Xj * beta_hat_j;
    sq_err_j = sum((y_hat_j - y).^2);       % squared error

    if sq_err_j < min_sq_err_j
        ft_index_j = j;
        min_sq_err_j = sq_err_j;
    end
end

min_sq_err_k = 1000;
ft_index_k = 0;

for k = 1:11                            % find third feature
    if k == ft_index_i || k == ft_index_j
        continue
    end

    Xk = [X(:, ft_index_i) X(:, ft_index_j) X(:, k)];          % concatenate 3 features together

    beta_hat_k = regress(y, Xk);     % perform regression on 3 features, this goes in table as betas
    y_hat_k = Xk * beta_hat_k;
    sq_err_k = sum((y_hat_k - y).^2);

    if sq_err_k < min_sq_err_k
        ft_index_k = k;
        min_sq_err_k = sq_err_k;
    end
end

beta_BS = zeros(11, 1);
beta_BS(ft_index_i) = beta_hat_k(1);
beta_BS(ft_index_j) = beta_hat_k(2);
beta_BS(ft_index_k) = beta_hat_k(3);

% Ridge
lambda = linspace(0, 1000, 100);
scaled = 1;     %0 means it's scaled

% outputs coefficient estimates
b_ridge = ridge(y, X, lambda, scaled);      % lambda is vector of the ridge parameters 
                                           
plot(lambda,b_ridge,'LineWidth',1)
grid on
xlabel('Ridge Parameter')
ylabel('Coefficient')
title('{\bf Ridge Trace}')
legend(ftrnames, 'Box', 'off', 'Location','best', 'Interpreter', 'none')


% Lasso
b_lasso = lasso(X, y, 'NumLambda', 20);

lassoPlot(b_lasso);
grid on
xlabel('Shrinkage Factor')
ylabel('Coefficient')
title('{\bf Lasso Plot}')
legend(ftrnames, 'Box', 'off', 'Location','best', 'Interpreter', 'none')

%format table 3
titlerow3 = {'LS'; 'Best_Subset'; 'Ridge'; 'Lasso'};
rowlabels3 = cell(ftrnames');

fprintf('The table below corresponds to Table 3.3 of the prostate example:');
fprintf('\n')
fprintf('\n')

tablearray = [beta_hat beta_BS b_ridge(:, 18) b_lasso(:, 18)];
table3 = array2table(tablearray, 'VariableNames', titlerow3, 'RowNames', rowlabels3);  %output table 3.1

%test_err3 = [sum((y_hat - y).^2)   % build test error row
%std_err3 = [std_err                % build standard error row
disp(table3)



