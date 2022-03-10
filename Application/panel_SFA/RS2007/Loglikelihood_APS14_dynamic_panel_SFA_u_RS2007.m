function logL = Loglikelihood_APS14_dynamic_panel_SFA_u_RS2007(theta, y, X, N, T, k, S, FMSLE_us)

    %tranform scale parameters back to true range
    theta(1) = exp(theta(1));
    theta(2:k) = 1./(1+exp(-theta(2:k))); %inverse logit transform of betas
    theta(k+3) = exp(theta(k+3));
    
    rhos = theta(k+4:end);
    Rho = inverse_mapping_vec(rhos); %Gaussian copula correlation matrix 
   
    alpha = theta(1);
    beta = theta(2:k);
    delta = theta(k+1:k+2);
    sigma2_v = theta(k+3);
      
    %Construct simulated_draws from the copula for u_{1}, ..., u_{T}
        %Cholesky decomp of the implied correlation matrix matrix
    [A,flag] = chol(Rho);
    if flag ~= 0
        logL = -sum(ones(N*T, 1)*-1e4); %Assign an arbitrarily large log density if Sigma is not positve definite
    else
        
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
            %transform simulated values to half-normal RV's
            CDF_u_i = normcdf(simulated_us_i*A, zeros(S, T), ones(S, T)); %Dependent uniform random variables from a Gaussian copula
            sigma2_u_hat = exp(delta(1) + delta(2).*(1:T));
            u_i = icdf('Half Normal', CDF_u_i, zeros(S, T), repmat(ones(1, T).*sqrt(sigma2_u_hat), S, 1)); % dependent half-normal RV's
            %adjust for possible unbalanced panel
            u_i = u_i(:,1:T-n_NaNs);
            %joint density
            den_i = mean(mvnpdf(rep_eps_i + u_i, zeros(1, T-n_NaNs), eye(T-n_NaNs).*sigma2_v)); %eq 1 pg. 510 section 5.1 APS14
            if (den_i < 1e-8) 
                den_i = 1e-8;
            end
            FMSLE_densities(i) = den_i; 
        end
        logL = -sum(log(FMSLE_densities));
    end
end