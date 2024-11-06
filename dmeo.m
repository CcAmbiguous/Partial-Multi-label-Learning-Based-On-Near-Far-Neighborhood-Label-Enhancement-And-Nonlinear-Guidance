%% The result is recorded in the result folder
clear
close all
dataname = 'emotions';
num_noise = 1;% The amount of noise added
% At present, there are two ways to add noise
% the first is to randomly add a certain amount of noise, the second is the average random noise, that is avg_cls.
% ttt = 1 the first way to add noise  
% ttt = 1 the second way to add noise
ttt = 1;  %Select your noise method, which defaults to the first method of adding noise

[~] = main_PML(dataname,num_noise,ttt);
% 
% [~] = main_PML('emotions',1,ttt);
% [~] = main_PML('emotions',2,ttt);
