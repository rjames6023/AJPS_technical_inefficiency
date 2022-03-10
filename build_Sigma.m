function Sigma = build_Sigma(n_inputs, rhos)
    %Covariance matrix (same as correlation matrix since we assume standard multivariate normal for P)
    Sigma_ = zeros(n_inputs,n_inputs);
    if n_inputs == 1
        Sigma = sqrt(rhos(1));
    elseif n_inputs == 2
        Sigma_triangular = [rhos(1), rhos(3)/(sqrt(rhos(1))*sqrt(rhos(2))), rhos(2)]; %triangular vector
        Sigma_(triu(ones(n_inputs,n_inputs))==1) = Sigma_triangular;
        Sigma = triu(Sigma_)+triu(Sigma_,1)';
    elseif n_inputs == 3
        Sigma_triangular = [rhos(1), rhos(4)/(sqrt(rhos(1))*sqrt(rhos(2))), rhos(2), rhos(5)/(sqrt(rhos(1))*sqrt(rhos(3))), rhos(6)/(sqrt(rhos(2))*sqrt(rhos(3))), rhos(3)]; %triangular vector
        Sigma_(triu(ones(n_inputs,n_inputs))==1) = Sigma_triangular;
        Sigma = triu(Sigma_)+triu(Sigma_,1)';
    end
    
end