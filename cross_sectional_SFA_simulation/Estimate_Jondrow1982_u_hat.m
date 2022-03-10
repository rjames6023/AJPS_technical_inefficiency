function [u_hat, V_u_hat] = Estimate_Jondrow1982_u_hat(theta, n_inputs, n_corr_terms, y, X)
    alpha = theta(1);
    beta = theta(2:n_inputs+1);
    sigma2_v = theta(2+n_inputs); 
    sigma2_u = theta(3+n_inputs);
    
    %Cobb-Douglas so alpha is logged
    obs_eps = y - log(alpha) - X*beta; %composed errors from the production function equation (i.e residuals from the production function)
    lambda = sqrt(sigma2_u/sigma2_v);
    sigma = sqrt(sigma2_u+sigma2_v);
    sig_star = sqrt(sigma2_u*sigma2_v/(sigma^2));
    u_hat = sig_star*(((normpdf(lambda.*obs_eps./sigma, 0, 1))./(1-normcdf(lambda.*obs_eps./sigma))) - ((lambda.*obs_eps)./sigma)); %Conditional distribution of u given eps. 
    V_u_hat = sig_star^2*(1+normpdf(lambda.*obs_eps./sigma)./(1-normcdf(lambda.*obs_eps./sigma)).*lambda.*obs_eps./sigma-(normpdf(lambda.*obs_eps./sigma)./(1-normcdf(lambda.*obs_eps./sigma))).^2);
end