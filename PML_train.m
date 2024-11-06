function model = PML_train(train_data, train_target,opt)

warning('off');
rng('default')
rng(1)

lambda1 = opt.lambda1;
lambda2 = opt.lambda2;
lambda3 = opt.lambda3;
lambda4 = opt.lambda4;
k = opt.k;
max_iter = opt.max_iter;
model = [];

[n,dim]=size(train_data);
[~,c]=size(train_target);

%% Training
X = [train_data, ones(n,1)];
J = train_target;  
d = dim + 1;
W = randn(d,c);
U = randn(k,c);
V = randn(d,k);
I = ones(n,k);
L = label_correlation(J);
miniLossMargin = 1e-3;
tic;%开始计算时间
for t = 1:max_iter
 
    %% linear
    F = ml_sigmoid(X*V)*U;
    % update W
    WA = 2*(X')*(X);
    WB = lambda1*eye(c)+0.5*(lambda2*L'+lambda2*L);
    WC = -X'*(F + J);
    W = lyap(WA, WB, WC);
    clear WA WB WC

    %% nonlinear
    H = ml_sigmoid(X*V);

    % update U
    U = pinv(lambda3*eye(k)+H'*H)*(H')*(X*W);

    % update V
    Q = exp(-X*V)./((I+exp(-X*V)).*(I+exp(-X*V)));
    grad_V = X'*(Q.*((H*U-X*W)*U')) + lambda4*V;

    V = V - 0.001*grad_V; 


    %%
    loss(t) = norm(X*W-J,'fro')^2+norm(X*W-ml_sigmoid(X*V)*U,'fro')^2+lambda1*norm(W,'fro')^2+lambda2*trace(W*L*W')+lambda3*norm(U,'fro')^2+lambda4*norm(V,'fro')^2;
    if t>5
        temp_loss = (loss(t-1) - loss(t))/loss(t-1); 
        if temp_loss<miniLossMargin
            break;%
        end
    end
    time = toc;%
end

model.W = W;
model.U = U;
model.V = V;
model.loss = loss;
model.time = time;
end

function Y = ml_sigmoid(X)
    I = ones(size(X));
    Y = I./(I+exp(-X));
end


