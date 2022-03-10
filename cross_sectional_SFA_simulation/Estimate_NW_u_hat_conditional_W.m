function [u_hat, V_u_hat] = Estimate_NW_u_hat_conditional_W(theta, n_inputs, n_corr_terms, y, X, P, U_hat, S_kernel)
    n = length(y);

    alpha = theta(1);
    beta = theta(2:n_inputs+1);
    sigma2_v = theta(3+n_inputs-1); 
    sigma2_u = theta(4+n_inputs-1);
    sigma2_w = theta(5+n_inputs-1:(5+n_inputs-1)+(n_inputs-2));
    
    %Observed variables
        %Cobb-Douglas so alpha is logged
    obs_eps = y - log(alpha) - X*beta; %composed errors from the production function equation (i.e residuals from the production function)
    W = (reshape(repmat(X(:,1), n_inputs-1, 1),n,n_inputs-1) - X(:,2:end)) - (P(:,2:end) - reshape(repmat(P(:,1), n_inputs-1, 1),n,n_inputs-1) + (log(beta(1)) - log(beta(2:end)))');
    rep_obs_eps = reshape(repelem(obs_eps, S_kernel), S_kernel, n);
    rep_obs_W = cell(1,n_inputs-1);
    for i=1:n_inputs-1
        w_i_rep = reshape(repelem(W(:,i), S_kernel),S_kernel, n);
        rep_obs_W(i) = {w_i_rep};
    end

    %Simulated variables
    simulated_v = normrnd(0, sqrt(sigma2_v), S_kernel, 1);
    simulated_u = sqrt(sigma2_u)*norminv((U_hat(:,1)+1)/2, 0,1); %simulated half normal rvs
    simulated_W = zeros(S_kernel, n_inputs-1);
    for i=1:n_inputs-1
        simulated_W(:,i) = norminv(U_hat(:,i+1), 0, sqrt(sigma2_w(i)));
    end
    simulated_eps = simulated_v - simulated_u; %Construct simulated eps (v-u)

    %Bandwidth information for each conditioning variable
    h_eps = 1.06*S_kernel^(-1/5)*(max(std(simulated_eps), iqr(simulated_eps)/1.34));
    h_W = zeros(1,n_inputs-1);
    for i=1:n_inputs-1
        h_W(i) = 1.06*S_kernel^(-1/5)*(max(std(simulated_W(:,i)), iqr(simulated_W(:,i))/1.34));
    end
    h = [h_eps, h_W];

    %kernel estimates for E[u|eps, w1, w2]
    %V[u|eps, w1, w2] = E[u^2|w2, w3, eps] - (E[u|w2, w3, eps])^2
    kernel_regression_results1 = zeros(n,1);
    kernel_regression_results2 = zeros(n,1);
    for i= 1:n
        eps_kernel = normpdf((simulated_eps - rep_obs_eps(:,i))./h(1));
        W_kernel = zeros(S_kernel,n_inputs-1);
        for j=1:n_inputs-1
            W_kernel(:,j) = normpdf((simulated_W(:,j) - rep_obs_W{j}(:,i))./h(j+1));
        end
        W_kernel_prod = prod(W_kernel, 2);
        kernel_product = eps_kernel.*W_kernel_prod;
        kernel_regression_results1(i,1) = sum(kernel_product.*simulated_u)/sum(kernel_product);
        kernel_regression_results2(i,1) = sum(kernel_product.*(simulated_u.^2))/sum(kernel_product);
    end
    u_hat = kernel_regression_results1;
    V_u_hat = kernel_regression_results2 - (u_hat.^2);
    
end