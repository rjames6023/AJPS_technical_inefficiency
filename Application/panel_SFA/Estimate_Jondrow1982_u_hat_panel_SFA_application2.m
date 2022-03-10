function [u_hat, V_u_hat] = Estimate_Jondrow1982_u_hat_panel_SFA_application2(alpha, beta, delta, sigma2_v, y, X, T, N)
    %For time verying sigma2_u
    
    %Cobb-Douglas so alpha is logged
    obs_eps = zeros(N, T);
    u_hat = zeros(N, T);
    V_u_hat = zeros(N, T);
    for t=1:T
        sigma2_u = exp(delta(1) + delta(2).*t);
        lambda = sqrt(sigma2_u)/sqrt(sigma2_v);
        sigma = sqrt(sigma2_u+sigma2_v);
        sig_star = sqrt(sigma2_u*sigma2_v/(sigma^2));
    
        u_hat_ = zeros(N, 1);
        V_u_hat_ = zeros(N, 1);
        obs_eps(:,t) = y(:,t) - log(alpha) - X{t}*beta'; %composed errors from the production function equation (i.e residuals from the production function)
        b = (obs_eps(:,t).*lambda)./sigma;
        u_hat_tmp = ((sigma*lambda)/(1 + lambda^2)).*(normpdf(b)./(1 - normcdf(b)) - b);
        V_u_hat_tmp = sig_star^2*(1+normpdf(b)./(1-normcdf(b)).*b-(normpdf(b)./(1-normcdf(b))).^2);

        u_hat_(1:length(u_hat_tmp), 1) = u_hat_tmp;
        V_u_hat_(1:length(V_u_hat_tmp), 1) = V_u_hat_tmp;
        if length(u_hat_tmp) < N
            u_hat_(length(u_hat_tmp):end) = NaN; %account for unbalanced panels
            V_u_hat_(length(V_u_hat_tmp):end) = NaN; %account for unbalanced panels
        end
        u_hat(:, t) = u_hat_;
        V_u_hat(:, t) = V_u_hat_;
    end
end