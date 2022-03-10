function logL = Loglikelihood_QMLE_cross_sectional_SFA(theta, y, X, P, n_inputs, n_corr_terms)
    n = length(y);

    alpha = exp(theta(1));
    beta = exp(theta(2:n_inputs+1));
    sigma2v = exp(theta(2+n_inputs));
    sigma2u = exp(theta(3+n_inputs)); 
    
    mu_W = theta(4+n_inputs:(4+n_inputs)+(n_inputs-2));
    Sigma_tril = theta(length(theta)+1-n_corr_terms:end);
    Sigma_hat = CholeskiToMatrix(Sigma_tril, n_inputs-1); 
    
    %Check if Sigma is positive definite
    [R,flag] = chol(Sigma_hat);
    if flag ~= 0
            logL = -sum(ones(n, 1)*-1e6); %Assign an arbitrarily large log density if Sigma is not positve definite
    else
        lambda = sqrt(sigma2u/sigma2v);
        sigma2 = sigma2u+sigma2v; 
        sigma = sqrt(sigma2);

        eps = y - log(alpha) - X*beta'; %composed errors from the production function equation (i.e residuals from the production function)
        W = (reshape(repmat(X(:,1), n_inputs-1, 1),n,n_inputs-1) - X(:,2:end)) - (P(:,2:end) - reshape(repmat(P(:,1), n_inputs-1, 1),n,n_inputs-1) + (log(beta(1)) - log(beta(2:end))));

        %Log density of the composed error
        den_eps = 2/sigma*normpdf(eps/sigma, 0, 1).*(1 - normcdf(lambda*eps/sigma, 0, 1)); %eq. (8) ALS77
        den_eps(den_eps < 1e-6) = 1e-6;
        logDen_eps = log(den_eps); 

        %Log density of the allocative inefficiency - multivariate normal density by assumption 
        den_alloc = mvnpdf(W, mu_W, Sigma_hat); %log multivariate normal density. by assumption the allocative inefficieny vector is multivariate normal.
        logDen_alloc = log(den_alloc);
        logDen = logDen_eps + logDen_alloc;
        logL = -sum(logDen); %negative log-likelihood
    end
end