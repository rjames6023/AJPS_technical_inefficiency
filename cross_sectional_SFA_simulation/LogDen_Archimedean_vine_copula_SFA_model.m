function logDen = LogDen_Archimedean_vine_copula_SFA_model(theta, n_inputs, n_corr_terms, y, X, P, us_Sxn, copula)
    n = length(y);
    [S,~] = size(us_Sxn); %number of simulated draws for evaluation of the integral

    alpha = theta(1);
    beta = theta(2:n_inputs+1);
    sigma2_v = theta(3+n_inputs-1); 
    sigma2_u = theta(4+n_inputs-1);
    sigma2_w = theta(5+n_inputs-1:(5+n_inputs-1)+(n_inputs-2));
    
    copula_dependence_params = theta(length(theta)-n_corr_terms+1:end); %last n_corr_terms elements are the copula dependence parameters
    %inverse transform the Gaussian copula correlation parameters
    rhos_log_form = copula_dependence_params(n_inputs:end);
    Rho = inverse_mapping_vec(rhos_log_form);
    
    [A,flag] = chol(Rho);
    if flag ~= 0
        logDen = ones(n, 1)*-1e4; %Assign an arbitrarily large log density if Rho is not positve definite
    else
        alpha21 = copula_dependence_params(1);
        if n_inputs == 3
            alpha31 = copula_dependence_params(2);
            rho32 = Rho(2,1);
        elseif n_inputs == 4
            alpha31 = copula_dependence_params(2);
            alpha41 = copula_dependence_params(3);
            rho32 = Rho(2,1);
            rho42 = Rho(3,1);
            rho43 = Rho(3,2);
        end

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
        %Compute the simulated integral over S simulated draws from the distribution of u (N+(half-normal))
        for j = 1:n
            CDF_w_j = zeros(S, n_inputs-1);
            for i=1:n_inputs-1
                CDF_w_j(:,i) = CDF_W_rep{i}(:,j);
            end
            %Copula pdf for obervation i over S simulated draws from N+(half-normal)
            if n_inputs == 2 %bivariate Archimedean copula density
                c21 = copulapdf(copula, [CdfUs(:,j), CDF_w_j], alpha21);
                simulated_copula_pdfs(:,j) = c21;
            elseif n_inputs == 3 %3-d vine copula construction with Archimedean and Gaussian bivariate copulas
                F_x2_conditional_x1 =  Archimedean_copula_partial_derivative_u2(copula, CDF_w_j(:,1), CdfUs(:,j), alpha21);
                F_x3_conditional_x1 = Archimedean_copula_partial_derivative_u2(copula, CDF_w_j(:,2), CdfUs(:,j), alpha31);
                c21 = copulapdf(copula, [CDF_w_j(:,1), CdfUs(:,j)], alpha21); 
                c31 = copulapdf(copula, [CDF_w_j(:,2), CdfUs(:,j)], alpha31); 
                c32_conditional_1 = copulapdf('Gaussian', [F_x3_conditional_x1, F_x2_conditional_x1], rho32); 
                simulated_copula_pdfs(:,j) = c21.*c31.*c32_conditional_1;
            elseif n_inputs == 4 %3-d vine copula construction with Archimedean and Gaussian bivariate copulas
                F_x2_conditional_x1 =  Archimedean_copula_partial_derivative_u2(copula, CDF_w_j(:,1), CdfUs(:,j), alpha21);
                F_x3_conditional_x1 = Archimedean_copula_partial_derivative_u2(copula, CDF_w_j(:,2), CdfUs(:,j), alpha31);
                F_x4_conditional_x1 = Archimedean_copula_partial_derivative_u2(copula, CDF_w_j(:,3), CdfUs(:,j), alpha41);

                F_x3_conditional_x1x2 = Gaussian_copula_partial_derivative_u2(F_x3_conditional_x1, F_x2_conditional_x1, rho32);
                F_x4_conditional_x1x2 = Gaussian_copula_partial_derivative_u2(F_x4_conditional_x1, F_x2_conditional_x1, rho42);

                c21 = copulapdf(copula, [CDF_w_j(:,1), CdfUs(:,j)], alpha21); 
                c31 = copulapdf(copula, [CDF_w_j(:,2), CdfUs(:,j)], alpha31); 
                c41 = copulapdf(copula, [CDF_w_j(:,3), CdfUs(:,j)], alpha41);
                c32_conditional_1 = copulapdf('Gaussian', [F_x3_conditional_x1, F_x2_conditional_x1], rho32);
                c42_conditional_1 = copulapdf('Gaussian', [F_x4_conditional_x1, F_x2_conditional_x1], rho42);
                c43_conditional_12 = copulapdf('Gaussian', [F_x4_conditional_x1x2, F_x3_conditional_x1x2], rho43);

                simulated_copula_pdfs(:,j) = c21.*c31.*c41.*c32_conditional_1.*c42_conditional_1.*c43_conditional_12;
            end
        end

        Integral = mean(simulated_copula_pdfs.*den_eps_plus_us)'; %Evaluation of the integral over S simulated samples. Column-wise mean.
        %Joint desnity. product of marginal density of w_{i}, i = 1, ..., n_inputs and the joint density f(\epsilon, w)
        prod_Den_W = prod(Den_W, 2);
        DenAll = prod_Den_W.*Integral; 
        logDen = log(DenAll);
    end
end