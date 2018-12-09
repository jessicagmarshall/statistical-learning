% Jessica Marshall & Jason Katz
% ECE411: Kaggle Contest SVM
% October 22, 2016

test = csvread('Kaggletest.csv');
test = normc(test);
data = csvread('Kaggletrain.csv');       %read in data
data = [data(:, 1) normc(data(:, 2:end))];     %normalize each column of the data, not the labels

y0 = datasample(1:50748, 390, 'Replace', false);        %randomly select 390 from label 0
y1 = datasample(50749:51147, 390, 'Replace', false);    % randomly select 390 from label 1
y = [y0 y1];

shuf = randperm(length(y));

for i = 1:length(shuf)          %randomly shuffle data points
    interleavedy(shuf(i)) = y(i);
end

% interleavedy = zeros(size(y));        %randomly shuffle instead
% interleavedy(1:2:end) = y0;
% interleavedy(2:2:end) = y1;

SVMdata = [];           %pick data minus the folds

for i = 1:780        
    SVMdata(i, :) = data(interleavedy(i), :);        %first column is the labels
end

X1 = SVMdata(1:702, 2:end);         %train
Y1 = SVMdata(1:702, 1);

X2 = SVMdata(703:end, 2:end);       %validate
Y2 = SVMdata(703:end, 1);

dataSet = prtDataSetClass(X1, Y1);     %create test set using Keene's thing
valSet = prtDataSetClass(X2, Y2);       % create validation set
test = prtDataSetClass(test);           % create test set
%plot(dataSet);

costvec = 2.*ones(1, 10);        %create vector of cost values
exp1 = linspace(-5, 20, 10);
costvec = costvec.^exp1;

gammavec = 2.*ones(1, 10);         %create vector of gamma values
exp2 = linspace(-15, 15, 10);
gammavec = gammavec.^exp2;

gridsearch = zeros(length(costvec), length(gammavec));     %create matrix to hold auc values

pcaSVMAlgorithm = prtPreProcPca('nComponents',10) + prtClassSvm;

for i = 1:length(costvec)
    pcaSVMAlgorithm.actionCell{2}.cost = costvec(i);
    
    for j = 1:length(gammavec)
        
        pcaSVMAlgorithm.actionCell{2}.gamma = gammavec(j);
        pcaSVMAlgorithm = pcaSVMAlgorithm.train(dataSet);
        yOut = pcaSVMAlgorithm.run(valSet);
        %prtScoreRoc(yOut, valSet);        %draw ROC curve
        gridsearch(i, j) = prtScoreAuc(yOut, valSet);

    end
end

[M, I] = max(gridsearch);
[M2, I2] = max(M);
cost_index = I(I2);
gamma_index = I2;
value = gridsearch(cost_index, gamma_index);      %value of max auc value; use to pick cost and gamma

pcaSVMAlgorithm.actionCell{2}.cost = costvec(cost_index);
pcaSVMAlgorithm.actionCell{2}.gamma = gammavec(gamma_index);

k = 10;        %# of folds
%pcaSVMAlgorithm = pcaSVMAlgorithm.train(dataSet);       %train using BEST cost and gamma values
[OUTPUTDATASET, TRAINEDACTIONS, CROSSVALKEYS] = pcaSVMAlgorithm.kfolds(dataSet, k);     % kfold using best algorithm

acuvec = zeros(1, k);

for i = 1:k
    yOut_val = TRAINEDACTIONS(k, 1).run(valSet);
    acuvec(i) = prtScoreAuc(yOut_val, valSet);
end

[M, I3] = max(acuvec);
pcaSVMAlgorithm_final = TRAINEDACTIONS(I3, 1);

%% TEST ON REAL DATA
yOut_test = pcaSVMAlgorithm_final.run(test);

col1 = linspace(1, 51147, 51147)';
col1 = [col1 yOut_test.data];
csvwrite('Kagglesubmit3.csv', col1, 1, 0);      %write to csv file to submit to Kaggle


