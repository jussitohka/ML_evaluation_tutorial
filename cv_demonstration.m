clear
addpath C:\Users\justoh\matlab\export_fig-master
dvec = 1:2:10;
nvec = 50:50:1000;
% values of deltavec depend on the dimensionality
% we will set the delta so that the bayes error rate is either 
% 0.1 or 0.05 
% BE = normcdf(-0.5*totaldelta),
% where totaldelta is the distance between the class means.
% Let delta be the distance between a single coordinate
% i.e. totaldelta = sqrt(d*delta^2) = sqrt(d)*delta
% -> delta = totaldelta/sqrt(d)
% deltavec stores total delta
deltavec = (-2)*[norminv(0.1) norminv(0.05)];

for dd = 1:length(dvec)
    d = dvec(dd);
    disp(dd)
    for nn = 1:length(nvec)
        n = nvec(nn);
        Ytrain = [zeros(n/2,1); ones(n/2,1)];
        Ytest = [zeros(5000,1); ones(5000,1)];
        fold = repmat([1:5]',n/5,1);
        for ee = 1:length(deltavec)
            delta = deltavec(ee)/sqrt(d);
            Xtest = randn(10000,d) + delta*Ytest;
            for iter = 1:1000
                Xtrain = randn(n,d) + delta*Ytrain;
                [cve{dd,nn,ee}(iter),ve{dd,nn,ee}(iter)] = cverror(Xtrain,Ytrain,fold);
                cls = classify(Xtest,Xtrain,Ytrain,'diagQuadratic');
                true_err{dd,nn,ee}(iter) = 1 - sum(cls == Ytest)/length(cls);
            end
            mcve(dd,nn,ee) = mean(cve{dd,nn,ee});
            mve(dd,nn,ee) = mean(ve{dd,nn,ee});
            scve(dd,nn,ee) = std(cve{dd,nn,ee});
            sve(dd,nn,ee) = std(ve{dd,nn,ee});
            maecve(dd,nn,ee) = mean(abs(true_err{dd,nn,ee} - cve{dd,nn,ee}));
            maeve(dd,nn,ee) = mean(abs(true_err{dd,nn,ee} - ve{dd,nn,ee}));
        end 
    end
end

addpath C:\Users\justoh\matlab\export_fig-master
close all
plot(nvec,maecve(1,:,2),'b')
hold
plot(nvec,maeve(1,:,2),'r')
plot(nvec,maecve(3,:,2),'k')
plot(nvec,maeve(3,:,2),'g')
plot(nvec,maecve(5,:,2),'c')
plot(nvec,maeve(5,:,2),'y')
legend('CV-5-d1','HO-d1','CV-5-d5','HO-d5','CV-5-d9','HO-d9')
xlabel('Mean absolute error of accuracy')
ylabel('Mean absolute error of accuracy')
xlabel('Number of samples')


export_fig C:\Users\justoh\Data\cv_vs_holdout_be5.png -r600

function [cve,ve] = cverror(Xtrain,Ytrain,fold)
         
err = zeros(max(fold),1);
for f = 1:max(fold)
    cls = classify(Xtrain(fold == f,:),Xtrain(fold ~= f,:),Ytrain(fold ~= f),'diagQuadratic');
    err(f) = 1 - sum(cls == Ytrain(fold == f))/length(cls);
end
cve = mean(err);
ve = err(1);
end
