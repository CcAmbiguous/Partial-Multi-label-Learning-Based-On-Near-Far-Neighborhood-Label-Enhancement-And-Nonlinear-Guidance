function [pLabels,data,target] = addnoise(dataname,num_noise,ttt)
%% Used to add noise and load the dataset
    load(dataname);
    tf1 = strcmp(dataname,'music_emotion');
    tf2 = strcmp(dataname,'music_style');
    tf3 = strcmp(dataname,'mirflickr');
    tf4 = strcmp(dataname,'YeastBP');
    tf = tf1+tf2+tf3+tf4;
        if tf == 1
            target = target';            
            if tf4 == 1
                pLabels = partial_labels';
            else
                pLabels = candidate_labels';
            end  
        else
            if ttt == 1
                [pLabels, noisy_nums] = rand_noisy_num(target,num_noise);
            end
            if ttt == 2
                [pLabels, noisy_nums] = rand_noisy_num_new(target,num_noise);
            end
        end

end