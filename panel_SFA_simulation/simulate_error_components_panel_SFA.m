function U_hat = simulate_error_components_panel_SFA(rhos, copula, T, S_kernel, seed, varargin)
    rng(seed)
    
    if strcmp(copula, 'Gaussian')
        %Estimated Gaussian copula correlation matrix.
        Rho_hat = inverse_mapping_vec(rhos);
        
        U_hat = copularnd('Gaussian', Rho_hat, S_kernel); %draw T dependent unfirm RV's for each s_{1}, ... S_kernel
        
    elseif strcmp(copula, 'students-t')
        nu_hat = rhos(1);
        Rho_hat = zeros(T,T);
        Rho_hat(itril(size(Rho_hat),-1)) = rhos(2:end);
        Rho_hat = Rho_hat + Rho_hat';
        Rho_hat = Rho_hat + eye(T);
        
        U_hat = copularnd('t', Rho_hat, nu_hat, S_kernel); %draw T dependent unfirm RV's for each s_{1}, ... S_kernel
        
    elseif strcmp(copula, 'vine')
        A = varargin{1};
        dvine_family_array = varargin{2};
        vine_copula_parameter_matrix = zeros(T-1, T-1);
        vine_copula_parameter_matrix(itril(size(vine_copula_parameter_matrix))) = rhos;
        vine_copula_parameter_array = num2cell(fliplr(vine_copula_parameter_matrix'));
        
        U_hat = simrvine(S_kernel, A, dvine_family_array, vine_copula_parameter_array, 1); %use parallel processing
    end
end