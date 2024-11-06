function [y_noisy, noisy_nums] = rand_noisy_num_new(y,noisy_num)
y_noisy=y;
[N,C]=size(y);
noisy_nums=zeros(N,1);
 for i=1:N
    u_idx=find(y(i,:)==0);
    truth_num = sum(y(i,:));
    U_num=length(u_idx);
    if truth_num < noisy_num && U_num >= (noisy_num - truth_num)
        rand_idx=randperm(U_num);
        rand_label= u_idx(rand_idx(1:(noisy_num - truth_num)));
        y_noisy(i,rand_label)=1;
        noisy_nums(i)=noisy_num;
    end
    if U_num <  (noisy_num - truth_num)
        y_noisy(i,u_idx)=1;
        noisy_nums(i)=U_num;
    end
end