function [y, X, Z, u, U] = Tran_Tsionas_simulation_setup(n, n_inputs, rho_u_W, true_gamma, true_alpha, true_beta, true_sigma2_u, true_sigma2_v, true_sigma2_w)
    %Tran and Tsionas setup
    u = abs(normrnd(0, sqrt(true_sigma2_u), n, 1));
    u_tilde = (u - mean(u))/std(u);
    W = zeros(n, n_inputs);
    CDF_W = zeros(n, n_inputs); %CDF of each w_i vector used to get empirical correlations for copula starting values 
    for i=1:n_inputs
        w_i = rho_u_W(i).*u_tilde + normrnd(0, 1, n, 1);
        W(:,i) = w_i;
        CDF_W(:,i) = normcdf(w_i, 0, sqrt(true_sigma2_w(i)));
    end
    
    v = normrnd(0, sqrt(true_sigma2_v), n, 1); %Random noise
    Z = chi2rnd(2, n, n_inputs); %Endogenous regressor/s
    X = zeros(n, n_inputs);
    for i=1:n_inputs
        X(:,i) = Z(:,i).*true_gamma + W(:,i); %Generate input based upon endogenous variables
    end
    y = true_alpha + X*true_beta' + (v - u); %simulated output
    U = [normcdf(u_tilde, 0, sqrt(true_sigma2_u)), CDF_W]; %Matrix of dependent random unfirom RVs 
end