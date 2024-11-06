function MicroF1 = Micro_F1(Pre_Labels,test_target)

[num_class,num_instance]=size(Pre_Labels);

TP=0;
FP=0;
FN=0;
for j=1:num_class
    temp=Pre_Labels(j,:).*test_target(j,:);
    TP=TP+sum(temp);
    FP=FP+sum(Pre_Labels(j,:))-sum(temp);
    FN=FN+sum(test_target(j,:))-sum(temp);
end
MicroF1=2*TP/(2*TP+FN+FP);
end

