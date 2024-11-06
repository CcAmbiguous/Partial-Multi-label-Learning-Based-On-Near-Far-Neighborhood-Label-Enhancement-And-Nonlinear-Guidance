function J = LENF(train_data,train_target,alpha,beta,mu)
% Near-far neighbor label enhancement
%% abstract
% alpha --- self weight [0,1]
% beta  --- far neighbor weight [0,1]
% mu    --- relative distance regulatory factor [0,1]
%% 
[N,~]=size(train_data);
[~,num_label]=size(train_target);
X = train_data;
Y = train_target;
J = Y;
S = knn_S(X);
c = ones(1,num_label);
%% label enhancement
for i = 1:N
    mn = max(S(i,:));
    mf = min(S(i,:));
    sigma = (mn-mf)*mu;
    tn = mn - sigma;
    tf = mf + sigma;

    a = zeros(1,num_label);
    b = zeros(1,num_label);
    num_n = 0;
    num_f = 0;
    for j = 2:N
        if S(i,j)>=tn
            num_n = num_n + 1;
            a = a + train_target(j,:);
        end
        if S(i,j)<=tf
            num_f = num_f + 1;
            b = b + (c-train_target(j,:));
        end
    end
    if num_n==0
        num_n = 1;
        a = c;
    end
    if num_f==0
        num_f = 1;
        b = c;
    end
    J(i,:) = alpha*Y(i,:)+(1-beta-alpha)*(a/num_n)+beta*(b/num_f);
end
%only the positive labels are enhanced
J = J.*Y;  %If you want to enhance the label globally, you can comment it out
end