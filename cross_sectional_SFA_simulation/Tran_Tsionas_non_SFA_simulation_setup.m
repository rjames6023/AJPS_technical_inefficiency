function [u, v, W] = Tran_Tsionas_non_SFA_simulation_setup(n, n_inputs, rho_u_W, true_sigma2_u, true_sigma2_v, true_sigma2_w, seed)
    rng(seed)
    %Tran and Tsionas setup
    u = abs(normrnd(0, sqrt(true_sigma2_u), n, 1));
    u_tilde = (u - mean(u))/std(u);
    W = zeros(n, n_inputs-1);
    for i=1:n_inputs-1
        w_i = rho_u_W(i).*u_tilde + normrnd(0, 1, n, sqrt(true_sigma2_w(i)));
        W(:,i) = w_i;
    end
    v = normrnd(0, sqrt(true_sigma2_v), n, 1); %Random noise
end