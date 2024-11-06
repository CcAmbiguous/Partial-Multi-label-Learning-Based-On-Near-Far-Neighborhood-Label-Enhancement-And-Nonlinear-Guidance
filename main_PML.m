function [result] = main_PML(dataname,num_noise,ttt)
% Initialization
% Fixed seed
rng('default');
rng(1);

addpath(genpath('datasets')); % Add path
addpath(genpath('function'));
addpath(genpath('metrics'));

if nargin < 3
    ttt = 1;
end
% Import data set
[pLabels,data,target] = addnoise(dataname,num_noise,ttt);
[N,c] = size(target);
%%
opt.lambda1 = 10;
opt.lambda2 = 1;
opt.lambda3 = 0.01;
opt.lambda4 = 10;
opt.k = c;% k <= c
opt.max_iter = 200;
%%
alpha = 0.7;     %--- self weight [0,1]
beta = 0.1;      %--- far neighbor weight [0,1]
mu = 0.2;        %--- relative distance regulatory factor [0,1]


draw = 0; % Whether to draw a loss chart, 1 is to draw a picture
indices = crossvalind('Kfold', 1:N ,10);  % Dividing the data set
result = {};
%%Ten fold cross verification
for round = 1:10
    ht = round*10;
    fprintf('%.1f%%\n',ht)
    test_idxs = (indices == round);                       
    train_idxs = ~test_idxs;                       
    train_data = data(train_idxs,:);                                           
    train_target = pLabels(train_idxs,:);                                                                                    
    test_data = data(test_idxs,:);                                          
    test_target = target(test_idxs,:);
                  
    % pre-processing                                       
    [train_data, settings]=mapminmax(train_data');                                        
    test_data=mapminmax('apply',test_data',settings);                                          
    train_data(isnan(train_data))=0;                                           
    test_data(isnan(test_data))=0;                                           
    train_data=train_data';                                           
    test_data=test_data';                                             
    X = train_data;
    Xt = test_data;
    Y = train_target;
    Yt = test_target;

    J = LENF(X,Y,alpha,beta,mu);

    % High dimensional kernel mapping 
    % [K,Kt] = Kernel_mapping(X',Xt');                       
    % X = K';   
    % Xt = Kt';
   
    %training
    model = PML_train(X,J,opt);
    time(round) = model.time;
    % %testing
    [HammingLoss(round),RankingLoss(round),OneError(round),Coverage(round),AveragePrecision(round),~] = PML_test(Xt,Yt,model);
end
fprintf('%s,num_noise=%.1f,λ1=%.3f,λ2=%.3f,λ3=%.3f,λ4=%.3f,k=%.1f\n HammingLoss=%.3f±%.3f\n RankingLoss=%.3f±%.3f\n OneError=%.3f±%.3f\n Coverage=%.3f±%.3f\n AveragePrecision=%.3f±%.3f\n', ...
    dataname,num_noise,opt.lambda1,opt.lambda2,opt.lambda3,opt.lambda4,opt.k,mean(HammingLoss),std(HammingLoss),mean(RankingLoss),std(RankingLoss),mean(OneError),std(OneError),mean(Coverage),std(Coverage),mean(AveragePrecision),std(AveragePrecision));
fprintf('oral_time=%.2f seconds\n',sum(time));
clear X Y Xt Yt data target train_data test_data train_target test_target 
filename = strcat('result/',dataname,'num_noise_',num2str(num_noise),'_predict.mat');
save(filename);
if draw == 1
    figure
    plot(1:length(model.loss),model.loss);
    title('Loss convergence diagram');
    xlabel('iter');
    ylabel('loss');;
end
result = {HammingLoss,RankingLoss,OneError,Coverage,AveragePrecision};
end