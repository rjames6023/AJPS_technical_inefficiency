function logL = Loglikelihood_Gaussian_copula_cross_sectional_application_SFA(theta, y, X, P, us_Sxn, n_inputs, n_corr_terms, S)
    n = length(y);
    
    alpha = exp(theta(1));
    beta = exp(theta(2:n_inputs+1));
    sigma2_v = exp(theta(2+n_inputs));
    sigma2_u = exp(theta(3+n_inputs)); 
    sigma2_w = exp(theta(4+n_inputs:(4+n_inputs)+(n_inputs-2)));
    mu_W = theta((4+n_inputs)+(n_inputs-2)+1:(4+n_inputs)+(n_inputs-2)+(n_inputs-1));

    rhos_log_form = theta(length(theta)+1-n_corr_terms:end);
    %Gaussian copula correlation matrix.
    Rho_hat = inverse_mapping_vec(rhos_log_form);
    [A,flag] = chol(Rho_hat);
    if flag ~= 0
        logL = -sum(ones(n, 1)*-1e4); %Assign an arbitrarily large log density if Rho is not positve definite
    else
        %Cobb-Douglas production function
        eps = y - log(alpha) - X*beta'; %composed errors from the production function equation (i.e residuals from the production function)
        W = (reshape(repmat(X(:,1), n_inputs-1, 1),n,n_inputs-1) - X(:,2:end)) - (P(:,2:end) - reshape(repmat(P(:,1), n_inputs-1, 1),n,n_inputs-1) + (log(beta(1)) - log(beta(2:end))));
        %Marginal density of allocative inefficiency terms
        Den_W = normpdf(W, repmat(mu_W,n,1), repmat(sqrt(sigma2_w),n,1));
        CDF_W = normcdf(W, repmat(mu_W,n,1), repmat(sqrt(sigma2_w),n,1));
    
        eps_Sxn = reshape(repelem(eps, S), S, n);
        us_Sxn_scaled = sqrt(sigma2_u)*us_Sxn;
        CdfUs = 2*(normcdf(sqrt(sigma2_u)*us_Sxn, 0, sqrt(sigma2_u)) -0.5);
        eps_plus_us = eps_Sxn + us_Sxn_scaled;
        den_eps_plus_us = normpdf(eps_plus_us, 0, sqrt(sigma2_v));
    
                %Evaluate the integral via simulation (to integrate out u from eps+u)
        simulated_copula_pdfs = zeros(S,n);
        CDF_W_rep = cell(1,n_inputs-1);
            %Compute the CDF (standard normal) for repeated allocative inefficiency terms
        for i=1:n_inputs-1
            CDF_W_rep(i) = {reshape(repelem(CDF_W(:,i), S),S, n)};
        end
        for j = 1:n
            CDF_w_j = zeros(S, n_inputs-1);
            for i=1:n_inputs-1
                CDF_w_j(:,i) = CDF_W_rep{i}(:,j);
            end
            c123 = copulapdf('Gaussian',[CdfUs(:,j), CDF_w_j], Rho_hat);
            simulated_copula_pdfs(:,j) = c123;
        end

        Integral = mean(simulated_copula_pdfs.*den_eps_plus_us)'; %Evaluation of the integral over S simulated samples. Column-wise mean.
        %Joint desnity. product of marginal density of w_{i}, i = 1, ..., n_inputs and the joint density f(\epsilon, w)
        prod_Den_W = prod(Den_W, 2);
        DenAll = prod_Den_W.*Integral; 
        DenAll(DenAll < 1e-6) = 1e-6;
        r = log(sum(beta));
        logDen = log(DenAll) + r;
        logL = -sum(logDen);
    end
end