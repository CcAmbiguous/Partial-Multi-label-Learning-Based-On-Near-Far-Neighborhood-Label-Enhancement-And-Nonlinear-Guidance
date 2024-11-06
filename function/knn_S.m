function P = knn_S(train_data)
%% Calculate sample similarity
[num_train, ~]=size(train_data);
distance = EuDist2(train_data,train_data,1);
[near_sample , ~] = sort(distance,2);
segma = sum(near_sample(:,2))/num_train;
P = exp(-distance/(2*segma^2));
end
