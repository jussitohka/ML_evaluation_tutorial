function [Xtrain, Xtest] = normalizeInput(Xtrain, Xtest)

m = mean(Xtrain, 1);
s = std(Xtrain, 1);

Xtrain = Xtrain - repmat(m, [size(Xtrain, 1), 1]);
Xtrain = Xtrain ./ repmat(s, [size(Xtrain, 1), 1]);

Xtest = Xtest - repmat(m, [size(Xtest, 1), 1]);
Xtest = Xtest ./ repmat(s, [size(Xtest, 1), 1]);

