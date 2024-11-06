function [XKTrain, XKTest] = Kernel_mapping(XTr, XTe)


    ntr = size(XTr,2);
    nte = size(XTe,2);
    if ntr<1000
        nAnchors = ntr;
    elseif ntr<5000
        nAnchors = ceil(ntr/2);
    elseif ntr<9000
        nAnchors = ceil(ntr/3);
    else
        nAnchors = ceil(ntr/4);
    end
    % nAnchors = ntr;
    % 随机选择锚点
    anchor_idx = randsample(ntr, nAnchors);
    Anchors = XTr(:,anchor_idx);
    % kmeans选锚点（备用）
%     [~, XAnchors] = litekmeans(XTrain, param.nXanchors, 'MaxIter', 30);
%     [~, YAnchors] = litekmeans(YTrain, param.nYanchors, 'MaxIter', 30);


    % 对训练数据核化
    XKTrain = sqdist_new(XTr,Anchors);
    Xsigma = mean(mean(XKTrain,2));
    XKTrain = exp(-XKTrain/(2*Xsigma));
    % 中心化
    Xmvec = mean(XKTrain);
    tmp = repmat(Xmvec,ntr,1);
    XKTrain = (XKTrain-tmp)';
    % 对测试数据核化
    XKTest = sqdist_new(XTe,Anchors);
    XKTest = exp(-XKTest/(2*Xsigma));
    XKTest = (XKTest-repmat(Xmvec,nte,1))';
end
function d=sqdist_new(a,b)
% 用于计算列向量间的相互关系
% SQDIST - computes squared Euclidean distance matrix
%          computes a rectangular matrix of pairwise distances
% between points in A (given in columns) and points in B

% NB: very fast implementation taken from Roland Bunschoten

aa = sum(a.^2,1); bb = sum(b.^2,1); ab = a'*b; 
d = (repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);


end