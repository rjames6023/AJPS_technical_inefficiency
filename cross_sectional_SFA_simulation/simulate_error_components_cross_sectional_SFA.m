function U_hat = simulate_error_components_cross_sectional_SFA(rho, copula, n_inputs, S_kernel, seed)
    rng(seed)
    
    if strcmp(copula, 'Gaussian')
        %Estimated Gaussian copula correlation matrix.
        Rho_hat = build_corr_matrix_u_w(n_inputs, rho(1:n_inputs-1), rho(n_inputs-1+1:end));
        U_hat = copularnd('Gaussian', Rho_hat, S_kernel); %S_kernel by n_inputs matrix of simulated error terms from gaussian copula (u,w1,w2). Returned as U(0,1) numbers
    
    elseif strcmp(copula, 'Clayton') || strcmp(copula, 'Gumbel') || strcmp(copula, 'Frank') %Bivariate Archimedian family
        if n_inputs == 2
            U_hat = copularnd(copula, rho, S_kernel);
        else
            if n_inputs == 3
                %Setup the vine copula parameters for Rosenblatt sampling
                    %Build vine structure matrix (tree structure of the vine copula)
                A = [1,1,1;0,2,2;0,0,3];
                    %Bivariate copula family matrix
                copula_families = {copula,copula; 'Gauss',0};
                    %Bivariate copula parameter matrix
                copula_params = {rho(1),rho(2); rho(3),0};  
            elseif n_inputs == 4
                %Setup the vine copula parameters for Rosenblatt sampling
                    %Build vine structure matrix (tree structure of the vine copula)
                A = [1,1,1,1; 0,2,2,2; 0,0,3,3; 0,0,0,4];
                    %Bivariate copula family matrix
                copula_families = {copula,copula,copula; 'Gauss','Gauss',0; 'Gauss',0,0};
                    %Bivariate copula parameter matrix
                copula_params = {rho(1),rho(2),rho(3); rho(4),rho(5),0; rho(6),0,0};  
            end
            %Simulate from the vine copula construction - uses the MatVines code - https://github.com/ElsevierSoftwareX/SOFTX-D-21-00039
            U_hat = simrvine(S_kernel, A, copula_families, copula_params, 1); %use parallel processing    
        end
    end
end