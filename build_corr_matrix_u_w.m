function Rho = build_corr_matrix_u_w(n_inputs, rho_u_W, rho_W)
    Rho_ = zeros(n_inputs,n_inputs);%Add 1 for technical inefficiency
    if n_inputs == 2
        Rho_triangular = [1, rho_u_W(1), 1]; %triangular vector
    elseif n_inputs == 3
        Rho_triangular = [1, rho_u_W(1), 1, rho_u_W(2), rho_W(1), 1]; %triangular vector
    elseif n_inputs == 4
        Rho_triangular = [1, rho_u_W(1), 1, rho_u_W(2), rho_W(1), 1, rho_u_W(3), rho_W(2), rho_W(3), 1]; %triangular vector
    end
    
    Rho_(triu(ones(n_inputs,n_inputs))==1) = Rho_triangular;
    %Correlation matrix
    Rho = triu(Rho_)+triu(Rho_,1)';
       
end