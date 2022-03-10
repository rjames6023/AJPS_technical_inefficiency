function logL = Loglikelihood_Archimedean_vine_copula_SFA_model(theta, n_inputs, n_corr_terms, y, X, P, us_Sxn, copula)
    %tranform scale parameters back to true range
    theta(1) = exp(theta(1));
    theta(2:n_inputs+1) = 1./(1+exp(-theta(2:n_inputs+1))); %inverse logit transform of betas
    theta(n_inputs+2:(n_inputs+1)+2+(n_inputs-1)) = exp(theta(n_inputs+2:(n_inputs+1)+2+(n_inputs-1))); %exponential transform of sigma2_u, sigma2_v and sigma2_W
    
    copula_dependence_params = theta(length(theta)-n_corr_terms+1:end); %last n_corr_terms elements are the copula dependence parameters
    copula_dependence_params(1:n_inputs-1) = exp(copula_dependence_params(1:n_inputs-1)); %exponential transform of bivariate unconditional Archimedian copula dependence params
    theta(length(theta)-n_corr_terms+1:end) = copula_dependence_params;
    
    %Obtain the log likelihood
    logDen = LogDen_Archimedean_vine_copula_SFA_model(theta, n_inputs, n_corr_terms, y, X, P, us_Sxn, copula);
    logL = -sum(logDen);
end