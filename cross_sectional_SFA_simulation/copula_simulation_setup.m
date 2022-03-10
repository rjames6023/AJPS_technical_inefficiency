function [y, X, Z, u, U] = copula_simulation_setup(n, n_inputs, Rho, true_gamma, true_alpha, true_beta, true_sigma2_u, true_sigma2_v, true_sigma2_w)
    U = copularnd('Gaussian', Rho, n);
    u = sqrt(true_sigma2_u)*norminv((U(:,1)+1)/2, 0, 1); %simulated half normal rvs
    %Allocative ineffieincy 
    if n_inputs == 1
        W = norminv(U(:,2:end), 0, sqrt(true_sigma2_w));
    else
        W = normpdf(U(:,2:end), repmat([0, 0],n,1), repmat(sqrt(true_sigma2_w),n,1));
    end
    
    v = normrnd(0, sqrt(true_sigma2_v), n, 1); %Random noise
    Z = chi2rnd(2, n, n_inputs); %Endogenous regressor/s
    X = zeros(n, n_inputs);
    for i=1:n_inputs
        X(:,i) = Z(:,i).*true_gamma + W(:,i); %Generate input based upon endogenous variables
    end
    y = true_alpha + X*true_beta' + (v - u); %simulated output
end