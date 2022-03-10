function logL = Loglikelihood_APS14_dynamic_panel_SFA_eps(theta, y, X, N, T, k)
    %tranform scale parameters back to true range
    theta(1) = exp(theta(1));
    theta(2:k) = 1./(1+exp(-theta(2:k))); %inverse logit transform of betas
    theta(k+1) = exp(theta(k+1));
    theta(k+2) = exp(theta(k+2));
    
    rhos = theta(k+3:end);
    Rho = inverse_mapping_vec(rhos);
    
    alpha = theta(1);
    beta = theta(2:k);
    sigma2_u = theta(k+1);
    sigma2_v = theta(k+2);
    
    %Additional params
    sigma2 = sigma2_u + sigma2_v;
    sigma = sqrt(sigma2);
    lambda = sqrt(sigma2_u)/sqrt(sigma2_v);
    
    eps = cell(T, 1);
    for t=1:T
        eps{t} = y{t} - log(alpha) - X{t}*beta';
    end
      
    %Compute log density of the margins
    logDen_eps_margins = cell(T,1);
    for t=1:T
        eps_p_den = ALS77_epsilon_density(eps{t}, sigma, lambda); %ALS77 composed error density 
        eps_p_logden = log(eps_p_den);
        logDen_eps_margins{t} = eps_p_logden;
    end
    
    %Compute F(eps) (CDF of epsilon) by integrating the density over the interval (-4, eps)
    eps_CDFs = cell(T,1);
    for t=1:T
        eps_CDFs{t} = zeros(N, 1);
        for i=1:N
            int = integral(@(eps) ALS77_epsilon_density(eps, sigma, lambda), -4, eps{t}(i)); %pdf for eps_it
            eps_CDFs{t}(i) = int;
        end
    end

    %Copula density for eps_1, ..., eps_T
    CDF_eps_matrix = [eps_CDFs{:}];
    copula_den = copulapdf('Gaussian', CDF_eps_matrix, Rho);
    log_copula_den = log(copula_den);
    
    logL = -sum((sum([logDen_eps_margins{:}], 2) + log_copula_den));
end