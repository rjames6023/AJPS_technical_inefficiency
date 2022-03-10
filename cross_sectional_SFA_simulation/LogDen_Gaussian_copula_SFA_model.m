function logDen = LogDen_Gaussian_copula_SFA_model(theta, n_inputs, n_corr_terms, y, X, P, us_Sxn)
    n = length(y);
    [S,~] = size(us_Sxn); %number of simulated draws for evaluation of the integral

    alpha = theta(1);
    beta = theta(2:n_inputs+1);
    sigma2_v = theta(2+n_inputs); 
    sigma2_u = theta(3+n_inputs);
    sigma2_w = theta(4+n_inputs:4+n_inputs+(n_inputs-1)-1);
    rhos_log_form = theta(length(theta)+1-n_corr_terms:end); %first n_inputs-1 entries are corr_u_W terms the rest are corr_W terms. In ascending order
  
    %Gaussian copula correlation matrix.
    Rho_hat = inverse_mapping_vec(rhos_log_form);
    [A,flag] = chol(Rho_hat);
    if flag ~= 0
        logDen = ones(n, 1)*-1e4; %Assign an arbitrarily large log density if Rho is not positve definite
    else

        %Cobb-Douglas production function
        eps = y - log(alpha) - X*beta; %composed errors from the production function equation (i.e residuals from the production function)
        W = (reshape(repmat(X(:,1), n_inputs-1, 1),n,n_inputs-1) - X(:,2:end)) - (P(:,2:end) - reshape(repmat(P(:,1), n_inputs-1, 1),n,n_inputs-1) + (log(beta(1)) - log(beta(2:end)))');
        %Marginal density of allocative inefficiency terms
        Den_W = normpdf(W, repmat(repelem(0, n_inputs-1),n,1), repmat(sqrt(sigma2_w'),n,1));
        CDF_W = normcdf(W, repmat(repelem(0, n_inputs-1),n,1), repmat(sqrt(sigma2_w'),n,1));

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
        logDen = log(DenAll);
    end
end