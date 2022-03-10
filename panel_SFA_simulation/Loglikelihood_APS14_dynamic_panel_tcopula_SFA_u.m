function logL = Loglikelihood_APS14_dynamic_panel_tcopula_SFA_u(theta, y, X, N, T, k, S)

    %tranform scale parameters back to true range
    theta(1) = exp(theta(1));
    theta(2:k) = 1./(1+exp(-theta(2:k))); %inverse logit transform of betas
    theta(k+1) = exp(theta(k+1));
    theta(k+2) = exp(theta(k+2));
    theta(k+3) = exp(theta(k+3));
    
    rhos = theta(k+4:end);
    Rho = inverse_mapping_vec(rhos); %Gaussian copula correlation matrix 
    nu = theta(k+3);
   
    alpha = theta(1);
    beta = theta(2:k);
    sigma2_u = theta(k+1);
    sigma2_v = theta(k+2);
    
    %Construct simulated_draws from the copula for u_{1}, ..., u_{T}
        %Cholesky decomp of the implied correlation matrix matrix
    [A,flag] = chol(Rho);
    if flag ~= 0
        logL = -sum(ones(N*T, 1)*-1e4); %Assign an arbitrarily large log density if Sigma is not positve definite
    else
                
        %simulated draws from the copula
        rng(123); %fix random seed for simulated likelihood
        u_draw = copularnd('t', Rho, nu, S);
        simulated_us = icdf('Half Normal', u_draw, zeros(S, T), ones(S, T).*sqrt(sigma2_u));
        FMSLE_us = cell(T, 1);
        for t=1:T
            us_Sxn = repmat(simulated_us(1:end, t)', N, 1)';
            FMSLE_us{t} = us_Sxn;
        end

        eps = cell(T, 1);
        for t=1:T
            eps__ = zeros(N, 1);
            tmp_eps = y{t} - log(alpha) - X{t}*beta'; %production function composed errors (epsilon)
            eps__(1:length(tmp_eps), 1) = tmp_eps;
            if length(tmp_eps) < N
                eps__(length(tmp_eps):end) = NaN; %account for unbalanced panels
            end
            eps{t} = eps__;
        end
    
        %Evaluate the integral via simulated MLE (FMSLE)
        all_eps = [eps{:}];
        FMSLE_densities = zeros(N, 1);
        for i=1:N
            eps_i = all_eps(i, 1:end);
            n_NaNs = length(eps_i(isnan(eps_i)));
            eps_i = eps_i(~isnan(eps_i)); %remove any NaN from an unbalanced panel
            rep_eps_i = repmat(eps_i, S, 1);
            simulated_us_i = zeros(S, T);
            for t = 1:T
               simulated_us_i(1:end, t) = FMSLE_us{t}(1:end, i);
            end
            %adjust for possible unbalanced panel
            simulated_us_i = simulated_us_i(:,1:T-n_NaNs);
            %joint density
            den_i = mean(mvnpdf(rep_eps_i + simulated_us_i, zeros(1, T-n_NaNs), eye(T-n_NaNs).*sigma2_v)); %eq 1 pg. 510 section 5.1 APS14
            if (den_i < 1e-6) 
                den_i = 1e-6;
            end
            FMSLE_densities(i) = den_i; 
        end
        logL = -sum(log(FMSLE_densities));
    end
end