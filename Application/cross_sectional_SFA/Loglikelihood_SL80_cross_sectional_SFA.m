function logL = Loglikelihood_SL80_cross_sectional_SFA(theta, y, X, P, n_inputs, n_corr_terms)
    n = length(y);

    alpha = exp(theta(1));
    beta = exp(theta(2:n_inputs+1));
    sigma2v = exp(theta(2+n_inputs));
    mu_W = theta(3+n_inputs:(3+n_inputs)+(n_inputs-2));
    mu_all = [0, mu_W];

    Sigma_tril = theta(length(theta)+1-(length(itril(n_inputs))):end);
    Sigma_hat = CholeskiToMatrix(Sigma_tril, n_inputs); 
    
    G = Sigma_hat;
    G(1,1) = G(1,1) + sigma2v; %eq. (A.3) from SL80
    Gstar = G;
    Gstar(1, 2:end) = -G(1, 2:end);
    Gstar(2:end, 1) = -G(2:end, 1);
    
    eps = y - log(alpha) - X*beta'; %composed errors from the production function equation (i.e residuals from the production function)
    W = (reshape(repmat(X(:,1), n_inputs-1, 1),n,n_inputs-1) - X(:,2:end)) - (P(:,2:end) - reshape(repmat(P(:,1), n_inputs-1, 1),n,n_inputs-1) + (log(beta(1)) - log(beta(2:end))));
    all = [eps W];
    r = sum(beta);
    InvSigma = inv(Sigma_hat);

    A = (eps + sigma2v*W*InvSigma(1, 2:end)')/sqrt(sigma2v)*sqrt(det(Sigma_hat)/det(G)); %eq. (A.5) from SL80
    Astar = (eps - sigma2v*W*InvSigma(1, 2:end)')/sqrt(sigma2v)*sqrt(det(Sigma_hat)/det(G)); %eq. (A.6) from SL80
    den = r*(normcdf(-A,0,1).*mvnpdf(all, mu_all, G) + normcdf(-Astar,0,1).*mvnpdf(all, mu_all, Gstar));
    den(den < 1e-6) = 1e-6;
    logden = log(den);
    logL = -sum(logden);
end