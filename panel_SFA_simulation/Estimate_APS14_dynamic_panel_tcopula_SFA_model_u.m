function [theta, logL] = Estimate_APS14_dynamic_panel_tcopula_SFA_model_u(theta0, X, y, N, T, k, S)
    
        Options = optimset('TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'off');
        [theta, logL] = fminunc(@(theta)Loglikelihood_APS14_dynamic_panel_tcopula_SFA_u(theta, y, X, N, T, k, S), theta0, Options);

        %tranform scale parameters back to true range
        theta(1) = exp(theta(1));
        theta(2:k) = 1./(1+exp(-theta(2:k))); %inverse logit transform of betas
        theta(k+1) = exp(theta(k+1));
        theta(k+2) = exp(theta(k+2));
        theta(k+3) = exp(theta(k+3));

        rhos = theta(k+4:end);
        Rho_hat = inverse_mapping_vec(rhos);
        theta(k+4:end) = Rho_hat(itril(size(Rho_hat), -1))';
        logL = logL*-1;
end