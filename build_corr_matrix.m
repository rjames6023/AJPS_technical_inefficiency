function Sigma = build_corr_matrix(n_inputs, P_rho)
    %Covariance matrix (same as correlation matrix since we assume standard multivariate normal for P)
    Sigma_ = zeros(n_inputs,n_inputs);
    if n_inputs == 1
        Sigma = 1;
    else
        if n_inputs == 2
            Sigma_triangular = [1, P_rho(1), 1]; %triangular vector
        elseif n_inputs == 3
            Sigma_triangular = [1, P_rho(1), 1, P_rho(2), P_rho(3), 1]; %triangular vector
        elseif n_inputs == 4
            Sigma_triangular = [1, P_rho(1), 1, P_rho(2), P_rho(3), 1, P_rho(4), P_rho(5), P_rho(6), 1]; %triangular vector
        end
        Sigma_(triu(ones(n_inputs,n_inputs))==1) = Sigma_triangular;
        Sigma = triu(Sigma_)+triu(Sigma_,1)';
    end
    
end