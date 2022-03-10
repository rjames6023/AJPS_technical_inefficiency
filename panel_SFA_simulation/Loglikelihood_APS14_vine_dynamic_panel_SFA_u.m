function logL = Loglikelihood_APS14_vine_dynamic_panel_SFA_u(theta, y, X, N, T, k, S, dvine_array, dvine_family_array)

    %tranform scale parameters back to true range
    theta(1) = exp(theta(1));
    theta(2:k) = 1./(1+exp(-theta(2:k))); %inverse logit transform of betas
    theta(k+1) = exp(theta(k+1));
    theta(k+2) = exp(theta(k+2));
    
    alpha = theta(1);
    beta = theta(2:k);
    sigma2_u = theta(k+1);
    sigma2_v = theta(k+2);
    
    copula_dependence_params = theta(k+3:end);
    if any(copula_dependence_params < -700, 'all') || any(copula_dependence_params > 700, 'all')
        logL = -sum(ones(N*T, 1)*-1e4); %Assign an arbitrarily large log density if any Gaussian copula parameter is outside bounds
    else 
        vine_copula_parameter_matrix = zeros(T-1, T-1);
        vine_copula_parameter_matrix(itril(size(vine_copula_parameter_matrix))) = copula_dependence_params;
        vine_copula_parameter_array = num2cell(fliplr(vine_copula_parameter_matrix'));
        
        eps = cell(T, 1);
        for t=1:T
            eps{t} = y{t} - log(alpha) - X{t}*beta'; %production function composed errors (epsilon)
        end
                
        %simulated draws from the copula
        rng(123); %fix random seed for simulated likelihood
        u_draw = simrvine(S, dvine_array, dvine_family_array, vine_copula_parameter_array, 1);
        simulated_us = icdf('Half Normal', u_draw, zeros(S, T), ones(S, T).*sqrt(sigma2_u));
        FMSLE_us = cell(T, 1);
        for t=1:T
            us_Sxn = repmat(simulated_us(1:end, t)', N, 1)';
            FMSLE_us{t} = us_Sxn;
        end

        %Evaluate the integral via simulated MLE (FMSLE)
        all_eps = [eps{:}];
        FMSLE_densities = zeros(N, 1);
        for i=1:N
            rep_eps_i = repmat(all_eps(i, 1:end), S, 1);
            simulated_us_i = zeros(S, T);
            for t = 1:T
               simulated_us_i(1:end, t) = FMSLE_us{t}(1:end, i);
            end
            %joint density
            den_i = mean(mvnpdf(rep_eps_i + simulated_us_i, zeros(1, T), eye(T).*sigma2_v)); %eq 1 pg. 510 section 5.1 APS14
            if (den_i < 1e-6) 
                den_i = 1e-6;
            end
            FMSLE_densities(i) = den_i; 
        end
        logL = -sum(log(FMSLE_densities));
    end
end