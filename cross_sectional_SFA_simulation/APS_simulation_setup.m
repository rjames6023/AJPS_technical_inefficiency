function [X, u, U] = APS_simulation_setup(n, n_inputs, Rho, beta, X_, r, sigma2_u, sigma2_v, sigma2_w, seed)
    rng(seed);
    %Simulate dependent (u,w1,...,wj)
    U = copularnd('Gaussian', Rho, n);
    u = sqrt(sigma2_u)*norminv((U(:,1)+1)/2, 0, 1); %half normal technical inefficiency
    %Allocative ineffieincy 
    W = norminv(U(:,2:end), repmat(repelem(0, n_inputs-1),n,1), repmat(sqrt(sigma2_w),n,1));
    v = normrnd(0, sqrt(sigma2_v), n, 1); %random noise error term 
    X = zeros(n, n_inputs);
    beta_r_rep = repmat(beta(2:end)./r,n,1);
    for i=1:n_inputs
        if i == 1
            X(:,i) = X_(:,i) - 1/r*(v-u) + sum(beta_r_rep.*W(:,1:end), 2); 
        else
            X(:,i) = X_(:,i) - 1/r*(v-u) - W(:,i-1) + sum(beta_r_rep.*W(:,1:end), 2); 
        end
    end
end