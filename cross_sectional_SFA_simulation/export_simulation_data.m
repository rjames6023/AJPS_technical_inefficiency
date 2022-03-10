function [] = export_simulation_data(theta, n_inputs, n_corr_terms, u, y, X, P, U_hat, S_kernel, simulation_number, filepath, export_label)
    n = length(y);
    
    alpha = theta(1);
    beta = theta(2:n_inputs+1);
    sigma2_v = theta(2+n_inputs); 
    sigma2_u = theta(3+n_inputs);
    sigma2_w = theta(4+n_inputs:4+n_inputs+(n_inputs-1)-1);

    %Observed variables
        %Cobb-Douglas so alpha is logged
    obs_eps = y - log(alpha) - X*beta; %composed errors from the production function equation (i.e residuals from the production function)
    W = (reshape(repmat(X(:,1), n_inputs-1, 1),n,n_inputs-1) - X(:,2:end)) - (P(:,2:end) - reshape(repmat(P(:,1), n_inputs-1, 1),n,n_inputs-1) + (log(beta(1)) - log(beta(2:end)))');

    %Simulated variables
    simulated_v = normrnd(0, sqrt(sigma2_v), S_kernel, 1);
    simulated_u = sqrt(sigma2_u)*norminv((U_hat(:,1)+1)/2, 0,1); %simulated half normal rvs
    simulated_W = zeros(S_kernel, n_inputs-1);
    for i=1:n_inputs-1
        simulated_W(:,i) = norminv(U_hat(:,i+1), 0, sqrt(sigma2_w(i)));
    end
    simulated_eps = simulated_v - simulated_u; %Construct simulated eps (v-u)
    
    %Export
    if ~exist(convertCharsToStrings([filepath filesep fullfile('Datasets','simulation_data',sprintf('n_inputs=%d', n_inputs),sprintf('n=%d',n),sprintf('%s',export_label))]))
        mkdir(convertCharsToStrings([filepath filesep fullfile('Datasets','simulation_data',sprintf('n_inputs=%d', n_inputs),sprintf('n=%d',n),sprintf('%s',export_label))]));
    end
    train_W_column_labels = cell(1, n_inputs-1);
    test_W_column_labels = cell(1, n_inputs-1);
    for i=1:n_inputs-1
        train_W_column_labels(i) = {sprintf('train_simulated_w%d', i)};
        test_W_column_labels(i) = {sprintf('train_simulated_w%d', i)};
    end
    NN_train_eps_W = array2table([simulated_eps, simulated_W]);
    NN_train_eps_W.Properties.VariableNames(:) = [{'train_simulated_eps'}, train_W_column_labels];
    NN_train_u = array2table(simulated_u);
    NN_train_u.Properties.VariableNames(:) = {'train_simulated_u'};
    NN_test_eps_W = array2table([obs_eps, W]);  
    NN_test_eps_W.Properties.VariableNames(:) = [{'test_simulated_eps'}, test_W_column_labels];
    NN_test_u = array2table(u);
    NN_test_u.Properties.VariableNames(:) = {'test_u'};
    
    writetable(NN_train_eps_W,[filepath filesep fullfile('Datasets','simulation_data',sprintf('n_inputs=%d', n_inputs),sprintf('n=%d',n),sprintf('%s',export_label),sprintf('SFA_simulation=%d_NN_train_eps_W.csv', simulation_number))]);
    writetable(NN_train_u,[filepath filesep fullfile('Datasets','simulation_data',sprintf('n_inputs=%d', n_inputs),sprintf('n=%d',n),sprintf('%s',export_label),sprintf('SFA_simulation=%d_NN_train_u.csv', simulation_number))]);
    writetable(NN_test_eps_W,[filepath filesep fullfile('Datasets','simulation_data',sprintf('n_inputs=%d', n_inputs),sprintf('n=%d',n),sprintf('%s',export_label),sprintf('SFA_simulation=%d_NN_test_eps_W.csv', simulation_number))]);
    writetable(NN_test_u,[filepath filesep fullfile('Datasets','simulation_data',sprintf('n_inputs=%d', n_inputs),sprintf('n=%d',n),sprintf('%s',export_label),sprintf('SFA_simulation=%d_NN_test_u.csv', simulation_number))]);
    fclose('all');
end