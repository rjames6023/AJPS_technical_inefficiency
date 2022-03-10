function [y, u] = panel_SFA_simulation_setup(N, T, rho1, X, true_alpha, true_beta, true_sigma2_v, true_sigma2_u, seed)
    
    rng(seed)
    pd_half_norm_transient = makedist('HalfNormal','mu',0,'sigma',sqrt(true_sigma2_u));

    v = normrnd(0, sqrt(true_sigma2_v), N*T, 1);
    
    u = zeros(N, T);
    y = zeros(N, T);

    j=1;
    for i=1:N
        u_i = zeros(T,1);
        for t=1:T
            if t ==1
                u_it = random(pd_half_norm_transient, 1);
            else
                u_it = rho1*u_i(t-1) + random(pd_half_norm_transient, 1); %Autoregressive (dynamic) technical inefficiency
            end
            u_i(t) = u_it;
            u(i,t) = u_it;
            y(i,t) = log(true_alpha) + true_beta*(X{t}(i, 1:end))' + v(j) - u_it;
            j = j + 1;
        end
    end
end