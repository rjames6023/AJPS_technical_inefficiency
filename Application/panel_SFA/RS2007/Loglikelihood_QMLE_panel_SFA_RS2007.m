function logL = Loglikelihood_QMLE_panel_SFA_RS2007(theta, y, X, N, T, k)

    %tranform scale parameters back to true range
    theta(1) = exp(theta(1));
    theta(2:k) = 1./(1+exp(-theta(2:k))); %inverse logit transform of betas
    theta(k+1) = exp(theta(k+1));
    theta(k+2) = exp(theta(k+2));
   
    alpha = theta(1);
    beta = theta(2:k);
    sigma2_u = theta(k+1);
    sigma2_v = theta(k+2);
    
    eps = cell(1,T);
    for t=1:T
        eps{t} = y{t} - log(alpha) - X{t}*beta'; %production function composed errors (epsilon)
    end
    
    cross_section_densities = cell(T, 1);
    lambda = sqrt(sigma2_u/sigma2_v); %Square root of the ratio of variance of technical and random errors.
    sigma2 = sigma2_u+sigma2_v; %Variance of the composed error term
    sigma = sqrt(sigma2);
    %sum of log ALS77 density over all i and t 
    for t = 1:T
        den_t = 2/sigma.*normpdf(eps{t}/sigma,0,1).*(1-normcdf((lambda*eps{t})/sigma,0,1)); %Density of the SFA model evaluated at the data
        cross_section_densities{t} = den_t;
    end
    all_den = vertcat(cross_section_densities{:});
    all_den(all_den < 1e-6) = 1e-6;
    logL = -sum(log(all_den));
end