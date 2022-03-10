function [] = export_finland_electricity_SFA_panel_data(theta, y, X, N, T, k, U_hat, S_kernel, filepath)
    alpha = theta(1);
    beta = theta(2:k);
    delta = theta(k+1:k+2);
    sigma2_v = theta(k+3);
  
    %Observed variables
    obs_eps = zeros(N, T);
    for t=1:T
        obs_eps(:,t) = y(:,t) - log(alpha) - X{t}*beta';
    end
    
    %Simulated variables
    simulated_v = mvnrnd(zeros(1,T), eye(T,T).*sigma2_v, S_kernel); %simulate random noise for all T panels 
    simulated_u = zeros(S_kernel, T);
    simulated_eps = zeros(S_kernel, T);
    for t=1:T
        sigma2_u = exp(delta(1) + delta(2)*t);
        simulated_u(:, t) = sqrt(sigma2_u)*norminv((U_hat(:,t)+1)/2, 0,1); %simulated half normal rvs
        simulated_eps(:,t) = simulated_v(:,t) - simulated_u(:,t);
    end
    
    train_eps_column_labels = cell(T,1);
    test_eps_column_labels = cell(T,1);
    train_u_column_labels = cell(T,1);
    for t=1:T
        train_eps_column_labels{t} = sprintf('train_eps_%d', t);
        test_eps_column_labels{t} = sprintf('test_eps_%d', t);   
        train_u_column_labels{t} = sprintf('train_u_%d', t);
    end
    
    NN_train_eps = array2table(simulated_eps);
    NN_train_eps.Properties.VariableNames(:) = train_eps_column_labels;
    NN_train_u = array2table(simulated_u);
    NN_train_u.Properties.VariableNames(:) = train_u_column_labels;
    NN_test_eps = array2table(obs_eps);  
    NN_test_eps.Properties.VariableNames(:) = test_eps_column_labels;
    
    writetable(NN_train_eps,[filepath filesep fullfile('panel_SFA_application_Results','panel_SFA_finnish_electricty_application_NN_train_eps.csv')]);
    writetable(NN_train_u,[filepath filesep fullfile('panel_SFA_application_Results','panel_SFA_finnish_electricty_application_NN_train_u.csv')]);
    writetable(NN_test_eps,[filepath filesep fullfile('panel_SFA_application_Results','panel_SFA_finnish_electricty_application_NN_test_eps.csv')]);
end