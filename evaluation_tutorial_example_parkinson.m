% a simple repeated CV example for the evaluation tutorial using naive
% Gaussian Bayes classifier
% **********************************************************
% This is a script to supplement the article:
% Jussi Tohka and Mark Van Gils: 
% "Evaluation of machine learning algorithms for Health and Wellness applications: 
% a tutorial", 2020
% **********************************************************
% Uses the data from :
% https://archive.ics.uci.edu/ml/datasets/Parkinson+Dataset+with+replicated+acoustic+features+#
% Naranjo, L., Perez, C.J., Campos-Roca, Y., Martin, J.: 
% Addressing voice recording replications for Parkinson's disease detection. 
% Expert Systems With Applications 46, 286-292 (2016)
% which contains acoustic features extracted from 3 voice recording replications of the sustained 
% /a/ phonation for each one of the 80 subjects (40 of them with Parkinson's Disease).
% IMPORTANT: 
% Each row can not be used independently, because is one of the three replications of one individual. 
% Nature of data is dependent for each subject, but independent from one to
% another subject. This must be taken into account when analyzing the data.
% The simplest way of averaging three measurements from each subject is used here
% to take care of the non-independence of the data. Other ways to tackle the issue can be 
% found in other demos based on this data.

clear
datadir = 'C:\Users\justoh\Data\evaluation_tutorial';
fn = 'ReplicatedAcousticFeatures-ParkinsonDatabase.csv';
data = csvread(fullfile(datadir,fn),1,1); % read only the numerical values
idx = 1:240; % use all the subjects
yyy = (data(idx,2) == 1);   % these are the labels (healthy control, PD subject) 
subjid = [1:80;1:80;1:80]; % generate the subject ids. Strictly speaking these
subjid = subjid(:);        % should read from the data file, but we will use a short 
subjid = subjid(idx);      % cut here
% we average (or sum) the three measurements per subject to avoid
% non-independendent train and test sets. The evaluation has always to be at the 
% subject level. note that we are here evaluating the diagnostic performance 
% of all three
% measurements averaged, so we need to have three measurements for each
% subject.
% note that we also remove the gender information as the data has been 
% collected using separate sampling and thus the gender ratios of PD subjects
% are not necessarily good approximations of the reality. 
for i = 1:80                   
    iii = find(subjid == i);
    x(i,:) = sum(data(iii,4:end));
    y(i) = yyy(iii(1)); 
end    
nfolds = 10; % set the number of folds in the CV
repeats = 25; % set the number of repeats of the repeated CV
% generate the folds, with stratification, i.e., each fold 
% contains equally many contorls and PD subjects  
for r = 1:25  
    foldid{r} = balanced_crossval(y,nfolds,[],0,0); % balanced_crossval function 
                                                    % is also available in
                                                    % this repository
end
yhat = zeros(length(y),25);
for r = 1:repeats % iterate over repeats
    for f = 1:nfolds % iterate over the folds
        % normalize the (training) data to have zero mean and unit variance
        % for each feature. The test data are nomalized based on means and
        % variances computed based on training data. Although no labels are
        % used in the normalization, this is a good practice. 
        [Xtrain, Xtest] = normalizeInput(x(foldid{r} ~=f,:),x(foldid{r} == f,:)); 
        yTrain = y(foldid{r} ~= f);
        yTest = y(foldid{r} == f);
        % train the NGB classifier using the training data, we can select
        % the empirical prior
        nb = fitcnb(Xtrain,yTrain,'DistributionNames','normal', 'Prior', 'empirical');
        % predict the labels and posterior probability of the test data
        % posteriors are needed later to compute the ROC curve and the AUC
        [label,posterior]=predict(nb,Xtest);
        yhat(foldid{r} == f,r) = posterior(:,2);
        % compute the sensitivity, specificity, accuracy, and balanced accuracy for each fold 
        % and the each repeat. Fold-wise results give important insight about the variability of the 
        % classification accuracy. Senspec function is also available in this repository
        [cvsen(r,f), cvspec(r,f), cvacc(r,f), cvbacc(r,f)] = senspec(yTest, label, 1);
    end
    % compute the total ensitivity, specificity, accuracy, and balanced
    % accuracy for each repeat
    % These are the same here as averaging fold-wise measures.
    [sen(r), spec(r), acc(r), bacc(r)] = senspec(y, yhat(:,r) > 0.5, 1);
    % compute still the AUC by the pooling method.
    [Xc{r},Yc{r},Tc{r},AUC(r)] = perfcurve(y,yhat(:,r),1);
end   