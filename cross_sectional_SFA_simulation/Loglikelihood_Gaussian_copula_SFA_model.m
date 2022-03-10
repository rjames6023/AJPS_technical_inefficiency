function logL = Loglikelihood_Gaussian_copula_SFA_model(theta, n_inputs, n_corr_terms, y, X, P, us_Sxn)
    %tranform scale parameters back to true range
    theta(1) = exp(theta(1));
    theta(2:n_inputs+1) = 1./(1+exp(-theta(2:n_inputs+1))); %inverse logit transform of betas
    theta(n_inputs+2:n_inputs+2+1+n_inputs-1) = exp(theta(n_inputs+2:n_inputs+2+1+n_inputs-1)); %exp transform of lsigma terms
    
    %Obtain the log likelihood
    logDen = LogDen_Gaussian_copula_SFA_model(theta, n_inputs, n_corr_terms, y, X, P, us_Sxn);
    logL = -sum(logDen);
end