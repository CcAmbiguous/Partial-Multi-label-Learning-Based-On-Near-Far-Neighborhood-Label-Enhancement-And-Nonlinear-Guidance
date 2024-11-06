function L = label_correlation(Y)
[~, class] = size(Y);
P = zeros(class, class);
for i=1:class
    for j=1:class
    
       if sum(Y(:,i))==0
           P(i,j) = 0;
       else    
       P(i,j) = (Y(:,i)'* Y(:,j))/(Y(:,i)'*Y(:,i));   
       end 
    end 
end
D = diag(sum(P,2));
L = D - P;
end
