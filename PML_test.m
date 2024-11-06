function [HammingLoss,RankingLoss,OneError,Coverage,AveragePrecision,MicroF1] = PML_test(test_data,test_target,model)

[num_test,~]=size(test_target);
[~,num_class]=size(test_target);
W = model.W;
X = [test_data,ones(num_test,1)];

Outputs = X*W;
[Outputs,~] = mapminmax(Outputs,0,1);
Pre_Labels = zeros(num_test,num_class);
Threshold = 0.8; %Prediction label binarization threshold
for i=1:num_test
    for k=1:num_class
        if(Outputs(i,k)>=0.9)
            Pre_Labels(i,k) = 1;
        else
            Pre_Labels(i,k) = 0;
        end
    end
end

HammingLoss=Hamming_loss(Pre_Labels,test_target);
RankingLoss=Ranking_loss(Outputs',test_target');
OneError=One_error(Outputs',test_target');
Coverage=coverage(Outputs',test_target');
AveragePrecision=Average_precision(Outputs',test_target');
MicroF1 = Average_precision(Pre_Labels',test_target');
end

