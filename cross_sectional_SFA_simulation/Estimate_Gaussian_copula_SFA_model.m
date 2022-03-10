function [theta, Logl] = Estimate_Gaussian_copula_SFA_model(theta0, n_inputs, n_corr_terms, y, X, P, us_Sxn)
    Options = optimset('TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'off');
%     gs = GlobalSearch('NumStageOnePoints', 1e3, 'NumTrialPoints', 1e3, 'Display', 'off');
%     problem = createOptimProblem('fmincon', 'x0', theta0, 'objective',@(theta)Loglikelihood_Gaussian_copula_SFA_model(theta, n_inputs, n_corr_terms, y, X, P, us_Sxn),'options', Options);
%     [theta, Logl] = run(gs, problem);
    [theta, Logl] = fminunc(@(theta)Loglikelihood_Gaussian_copula_SFA_model(theta, n_inputs, n_corr_terms, y, X, P, us_Sxn), theta0, Options);

    %tranform scale parameters back to true range
    theta(1) = exp(theta(1));
    theta(2:n_inputs+1) = 1./(1+exp(-theta(2:n_inputs+1))); %inverse logit transform of betas
    theta(n_inputs+2:n_inputs+2+1+n_inputs-1) = exp(theta(n_inputs+2:n_inputs+2+1+n_inputs-1)); %exp transform of lsigma terms
    
    %Gaussian copula correlation matrix parameters
    rhos_log_form = theta(length(theta)+1-n_corr_terms:end);
    Rho = inverse_mapping_vec(rhos_log_form);
    theta(length(theta)+1-n_corr_terms:end) = Rho(itril(size(Rho), -1))';
    Logl = Logl*-1;
end