function u_hat = Estimate_NW_u_hat_panel_SFA(theta, y, X, N, T, k, U_hat, S_kernel)
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

    %Bandwidth information for each conditioning variable
    h_eps = zeros(T, 1);
    for t=1:T
        h_eps(t) = 1.06*S_kernel^(-1/5)*(max(std(simulated_eps(:,t)), iqr(simulated_eps(:,t))/1.34));
    end

    %kernel estimates for E[u_{t}|eps_{t}]
    kernel_regression_results1 = zeros(N, T);
    for i=1:N 
        panel_i_kernel_regression_results = zeros(T, 1);
        for t=1:T
            kernel_product = normpdf((simulated_eps(:,t) - obs_eps(i,t))./h_eps(t));
            panel_i_kernel_regression_results(t) = sum(kernel_product.*simulated_u(:,t))/sum(kernel_product);
        end
        kernel_regression_results1(i, :) = panel_i_kernel_regression_results';
    end
    u_hat = kernel_regression_results1; 
end