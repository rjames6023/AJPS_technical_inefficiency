function [u_hat, V_u_hat] = Estimate_NW_u_hat_conditional_eps_panel_SFA(theta, y, X, N, T, k, U_hat, S_kernel)
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

    %kernel estimates for E[u_{t}|eps_{1}, ..., eps_{T}]
    %V[u|eps, w1, w2] = E[u^2|w2, w3, eps] - (E[u|w2, w3, eps])^2
    kernel_regression_results1 = zeros(N, T);
    kernel_regression_results2 = zeros(N, T);
    for i=1:N 
        panel_i_kernel_regression_results = zeros(T, 1);
        panel_i_kernel_regression_results2 = zeros(T, 1);
        eps_kernel = zeros(S_kernel, T);
        %Construct the kernel distances for all T time periods
        for t=1:T
            eps_kernel(:,t) = normpdf((simulated_eps(:,t) - obs_eps(i, t))./h_eps(t));
        end
        kernel_product = prod(eps_kernel, 2);
        for j=1:T %NW for each t = 1, .., T observation in each panel i
            panel_i_kernel_regression_results(j) = sum(kernel_product.*simulated_u(:, j))/sum(kernel_product);
            panel_i_kernel_regression_results2(j) = sum(kernel_product.*simulated_u(:,j).^2)/sum(kernel_product);
        end
        kernel_regression_results1(i, :) = panel_i_kernel_regression_results';
        kernel_regression_results2(i,  :) = panel_i_kernel_regression_results2';
    end
    u_hat = kernel_regression_results1; 
    u_hat2 = kernel_regression_results2; 
    V_u_hat = u_hat2 - (u_hat.^2);
end