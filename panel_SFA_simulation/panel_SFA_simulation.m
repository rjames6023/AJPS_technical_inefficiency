function panel_SFA_simulation(N, T, rho1, copula)  
    %filepath = 'C:\Users\Robert James\Dropbox (Sydney Uni)\Estimating Technical Inefficiency Project';
    filepath = [filesep fullfile('project','RDS-FOB-Eff_Dens_1-RW','technical_inefficiency_estimation')]; %works on windows and linux
    addpath([filepath filesep fullfile('Code')])
    
    column_labels = {'AR_1', 'JLMS', 'NW', 'NW_conditional_eps'};
    export_label = sprintf('rho=%.2f',rho1);
    %Simulation parameters
    num_iterations = 500;
    S = 500; %number of random draws used to evaluate the simulated likelihood (FMSLE)
    S_kernel = 10000; %number of simulated draws for evaluation of the conditional expectation
    
    %Simulation parameters
    true_alpha = 1;
    true_beta = [0.5, 0.3];
    true_sigma2_v = 0.25;
    true_sigma2_u = 1;
    
    k = length(true_beta) + 1; %number of production factor regressors + constant + pi
    n_inputs = length(true_beta);
    n_corr_terms = ((T)^2-(T))/2; 
    
    rng(1234)
    X_mu = [0, 0];
    X_Sigma = [1.5, 0; 0, 1];
    X_tilde = mvnrnd(X_mu, X_Sigma, N*T);
    X = cell(T, 1);  %Build cell array X where each cell is a N by J matrix of all cross sections at time t
    for t=1:T 
        if t == 1
            X{t} = X_tilde(t:N, 1:end); 
        else
            X{t} = X_tilde((t-1)*N+1:t*N, 1:end);
        end
    end

    if ~exist(convertCharsToStrings([filepath filesep fullfile('panel_SFA_simulation_Results',copula,sprintf('AR_1=%.2f', rho1))]))
        mkdir(convertCharsToStrings([filepath filesep fullfile('panel_SFA_simulation_Results',copula,sprintf('AR_1=%.2f', rho1))]));
    end
    
    %%
    %Matrices to hold MSE results.
    Logl_matrix = zeros(num_iterations, 1+1); %Column is Log-likelihood
    %Columns are: Jondrow1982, NW_u_hat, NW_u_hat_conditional_w1w2 
    u_hat_MSE_matrix = zeros(num_iterations, 1+3); 
    u_hat_upper_quartile_MSE_matrix = zeros(num_iterations, 1+3); 
    u_hat_mid_quartile_MSE_matrix = zeros(num_iterations, 1+3); 
    u_hat_lower_quartile_MSE_matrix = zeros(num_iterations, 1+3); 
    std_u_hat_matrix = zeros(num_iterations, 1+3); 
       
    %Matrices to hold Murphy Diagram score results.   
    JLMS_Murphy_scores_matrix = zeros(501, num_iterations);
    NW_Murphy_scores_matrix = zeros(501, num_iterations);
    NW_conditional_W_Murphy_scores_matrix = zeros(501, num_iterations);
    
    %%
    for c=1:num_iterations
        seed = c;
        rng(seed)
        
        %Simulate sample
        [y, u] = panel_SFA_simulation_setup(N, T, rho1, X, true_alpha, true_beta, true_sigma2_v, true_sigma2_u, seed);

        eps = zeros(N, T);
        for t=1:T
            eps(:,t) = y(:,t) - log(true_alpha) - X{t}*true_beta'; %production function composed errors (epsilon)
        end
         
        %Estimate APS14 copula based dynamic SFA model
            %initial values for log-likelihood maximumization
        initial_sigma2_v = true_sigma2_v;
        initial_lsigma2_v = log(initial_sigma2_v);
%         initial_sigma2_u = true_sigma2_u;
%         initial_lsigma2_u = log(initial_sigma2_u);
        initial_delta = [log(true_sigma2_u), rho1/2];
        initial_alpha = 1;
        initial_lalpha = log(initial_alpha);
        initial_beta = true_beta;
        initial_logit_beta = log(initial_beta./(1-initial_beta));
        
        if strcmp(copula, 'Gaussian')
            %Independent uniform random variables for FMSLE - assumes a Gaussian/t copula
            p = sobolset(T);
            p_scramble = scramble(p,'MatousekAffineOwen');
            FMSLE_us = cell(T, 1);
            us_ = net(p_scramble, S);
            us_ = norminv(us_, zeros(S, T), ones(S, T)); %transform to standard normal 
            for t=1:T
                us_Sxn = repmat(us_(1:end, t)', N, 1)';
                FMSLE_us{t} = us_Sxn;
            end
                %initial values for Gaussian copula
            sigma2_u = exp(initial_delta(1) + initial_delta(2).*(1:T));
            PIT_u = cdf('Half Normal', u, zeros(N, T), repmat(ones(1, T).*sqrt(sigma2_u), N, 1));
            norm_inv_u = norminv(PIT_u, zeros(N, T), ones(N, T));
            initial_Rho = corr(norm_inv_u, 'Type','Kendall');
            initial_lRho = direct_mapping_mat(initial_Rho); 

            theta0 = [initial_lalpha, initial_logit_beta, initial_delta, initial_lsigma2_v, initial_lRho'];
            [theta, APS14_logL] = Estimate_APS14_dynamic_panel_SFA_model_u(theta0, X, y, N, T, k, S, FMSLE_us);
            APS14_logL = APS14_logL*-1; %flip the negative log-likelihood
            
        elseif strcmp(copula, 'students-t')
            %Independent uniform random variables for FMSLE - assumes a Gaussian/t copula
            p = sobolset(T);
            p_scramble = scramble(p,'MatousekAffineOwen');
            FMSLE_us = cell(T, 1);
            us_ = net(p_scramble, S);
            us_ = norminv(us_, zeros(S, T), ones(S, T)); %transform to standard normal 
            for t=1:T
                us_Sxn = repmat(us_(1:end, t)', N, 1)';
                FMSLE_us{t} = us_Sxn;
            end
            %initial values for students-t copula
            PIT_u = cdf('Half Normal', [u{:}], zeros(N, T), ones(N, T).*sqrt(true_sigma2_u));
            [initial_Rho,initial_nu] = copulafit('t', PIT_u);
            initial_lRho = direct_mapping_mat(initial_Rho); 
            initial_lnu = log(initial_nu);
            theta0 = [initial_lalpha, initial_logit_beta, initial_lsigma2_u, initial_lsigma2_v, initial_lnu, initial_lRho'];
            [theta, APS14_logL] = Estimate_APS14_dynamic_panel_tcopula_SFA_model_u(theta0, X, y, N, T, k, S);
            APS14_logL = APS14_logL*-1; %flip the negative log-likelihood
            
        elseif strcmp(copula, 'vine')
           %Setup the vine copula construction matrices       
            dvine_array = cdvinearray('d', T); %d-vine copula array
            dvine_family_array = cell(T-1, T-1);
            for i=1:T-1
                for j=1:T-1
                    if i <= j
                        dvine_family_array{i,j} = 'frank';
                    end
                end
            end
            dvine_family_array = fliplr(dvine_family_array);
            %Fit Vine decomposition SFA model
            initial_Rho = corr([u{:}], 'Type','Kendall');
            dvine_initial_parameter_matrix = zeros(T-1, T-1); %Copula parameter matrix 
            %Loop over all bivariate copula pairs for a D-vine copula
            for j=1:T-1
                for i=1:T-j
                    dvine_initial_parameter_matrix(j, i) = copulaparam('frank', initial_Rho(i, (i+j)));
                end
            end
            flipped_transposed_dvine_initial_parameter_matrix = fliplr(dvine_initial_parameter_matrix)';
            dvine_initial_parameter_vector = flipped_transposed_dvine_initial_parameter_matrix(itril(size(flipped_transposed_dvine_initial_parameter_matrix)));
            theta0 = [initial_lalpha, initial_logit_beta, initial_lsigma2_u, initial_lsigma2_v, dvine_initial_parameter_vector'];
            [theta, APS14_logL] = Estimate_APS14_vine_dynamic_panel_SFA_model_u(theta0, X, y, N, T, k, S, dvine_array, dvine_family_array);
            APS14_logL = APS14_logL*-1; %flip the negative log-likelihood
        end
        
        %%
        %Simulated dependent U based upon estimated copula parameters
        if strcmp(copula, 'Gaussian')
            U_hat = simulate_error_components_panel_SFA(theta(k+4:end), 'Gaussian', T, S_kernel, seed);
        elseif strcmp(copula, 'students-t')
            U_hat = simulate_error_components_panel_SFA(theta(k+4:end), 'students-t', T, S_kernel, seed);
        elseif strcmp(copula, 'vine')
            U_hat = simulate_error_components_panel_SFA(theta(k+4:end), 'vine', T, S_kernel, seed, dvine_array, dvine_family_array);
        end
        
        %Estimate the technical ineffieincy based upon ML solution
        JLMS_u_hat = Estimate_Jondrow1982_u_hat_panel_SFA(theta, y, X, k, T, N);
        %Technical inefficiency using information from the joint distribution
        NW_u_hat = Estimate_NW_u_hat_panel_SFA(theta, y, X, N, T, k, U_hat, S_kernel);
        NW_u_hat_conditional_eps = Estimate_NW_u_hat_conditional_eps_panel_SFA(theta, y, X, N, T, k, U_hat, S_kernel); %Multivariate Nadaraya Watson non-parametric estimator conditional on epsilons
        
        %Scores for the Murphy Diagrams
        [JLMS_S1_avg, NW_S2_avg] = Murphy_Diagram_scores(vertcat(u(:)), reshape(JLMS_u_hat,1,[])', reshape(NW_u_hat,1,[])', true_sigma2_u);
        [~, NW_conditional_W_S2_avg] = Murphy_Diagram_scores(vertcat(u(:)), reshape(JLMS_u_hat,1,[])', reshape(NW_u_hat_conditional_eps,1,[])', true_sigma2_u);
        
        JLMS_Murphy_scores_matrix(:,c) = JLMS_S1_avg;
        NW_Murphy_scores_matrix(:,c) = NW_S2_avg;
        NW_conditional_W_Murphy_scores_matrix(:,c) = NW_conditional_W_S2_avg;
        
        %Export simulated datasets for NW hyper-parameter learning and additional modelling of external non-parametric regression models
        export_SFA_panel_simulation_data(theta, y, X, u, N, T, k, U_hat, S_kernel, c, filepath, export_label);
        
        %%
        %Compute MSE of technical ineffieincy estimates
        Jondrow1982_u_hat_MSE = (sum((u - JLMS_u_hat).^2, 'all'))/(N*T);
        NW_u_hat_MSE = (sum((u - NW_u_hat).^2, 'all'))/(N*T);
        NW_u_hat_conditional_eps_MSE = (sum((u - NW_u_hat_conditional_eps).^2, 'all'))/(N*T);
        
        Jondrow1982_upper_quartile_u_hat_MSE = zeros(T,1);
        NW_u_hat_upper_quartile_u_hat_MSE = zeros(T, 1);
        NW_u_hat_conditional_eps_upper_quartile_u_hat_MSE = zeros(T, 1);
        
        Jondrow1982_lower_quartile_u_hat_MSE = zeros(T, 1);
        NW_u_hat_lower_quartile_u_hat_MSE = zeros(T, 1);
        NW_u_hat_conditional_eps_lower_quartile_u_hat_MSE = zeros(T, 1);

        Jondrow1982_mid_quartile_u_hat_MSE = zeros(T, 1);
        NW_u_hat_mid_quartile_u_hat_MSE = zeros(T, 1);
        NW_u_hat_conditional_eps_mid_quartile_u_hat_MSE = zeros(T, 1);

        for t=1:T
         idx_upper = find(u(:,t) >= quantile(u(:,t), 0.75));
         Jondrow1982_upper_quartile_u_hat_MSE(t) = mean((u(idx_upper,t) - JLMS_u_hat(idx_upper,t)).^2);
         NW_u_hat_upper_quartile_u_hat_MSE(t) = mean((u(idx_upper, t) - NW_u_hat(idx_upper, t)).^2);
         NW_u_hat_conditional_eps_upper_quartile_u_hat_MSE(t) = mean((u(idx_upper, t) - NW_u_hat_conditional_eps(idx_upper, t)).^2);

         idx_lower = find(u(:,t) <= quantile(u(:,t), 0.25));
         Jondrow1982_lower_quartile_u_hat_MSE(t) = mean((u(idx_lower, t) - JLMS_u_hat(idx_lower, t)).^2);
         NW_u_hat_lower_quartile_u_hat_MSE(t) = mean((u(idx_lower, t) - NW_u_hat(idx_lower, t)).^2);
         NW_u_hat_conditional_eps_mid_quartile_u_hat_MSE(t) = mean((u(idx_lower, t) - NW_u_hat_conditional_eps(idx_lower, t)).^2);

         idx_mid = find((u(:,t) > quantile(u(:,t), 0.25)) & (u(:,t) < quantile(u(:,t), 0.75)));
         Jondrow1982_mid_quartile_u_hat_MSE(t) = mean((u(idx_mid, t) - JLMS_u_hat(idx_mid, t)).^2);
         NW_u_hat_mid_quartile_u_hat_MSE(t) = mean((u(idx_mid, t) - NW_u_hat(idx_mid, t)).^2);
         NW_u_hat_conditional_eps_lower_quartile_u_hat_MSE(t) = mean((u(idx_mid, t) - NW_u_hat_conditional_eps(idx_mid, t)).^2);
        end
        
        Jondrow1982_upper_quartile_u_hat_MSE = mean(Jondrow1982_upper_quartile_u_hat_MSE);
        NW_u_hat_upper_quartile_u_hat_MSE = mean(NW_u_hat_upper_quartile_u_hat_MSE);
        NW_u_hat_conditional_eps_upper_quartile_u_hat_MSE = mean(NW_u_hat_conditional_eps_upper_quartile_u_hat_MSE);
        
        Jondrow1982_lower_quartile_u_hat_MSE = mean(Jondrow1982_lower_quartile_u_hat_MSE);
        NW_u_hat_lower_quartile_u_hat_MSE = mean(NW_u_hat_lower_quartile_u_hat_MSE);
        NW_u_hat_conditional_eps_lower_quartile_u_hat_MSE = mean(NW_u_hat_conditional_eps_lower_quartile_u_hat_MSE);

        Jondrow1982_mid_quartile_u_hat_MSE = mean(Jondrow1982_mid_quartile_u_hat_MSE);
        NW_u_hat_mid_quartile_u_hat_MSE = mean(NW_u_hat_mid_quartile_u_hat_MSE);
        NW_u_hat_conditional_eps_mid_quartile_u_hat_MSE = mean(NW_u_hat_conditional_eps_mid_quartile_u_hat_MSE);
        
        u_hat_MSE_matrix(c,2:end) = [Jondrow1982_u_hat_MSE, NW_u_hat_MSE, NW_u_hat_conditional_eps_MSE]; 
        u_hat_upper_quartile_MSE_matrix(c, 2:end) = [Jondrow1982_upper_quartile_u_hat_MSE, NW_u_hat_upper_quartile_u_hat_MSE, NW_u_hat_conditional_eps_upper_quartile_u_hat_MSE];
        u_hat_mid_quartile_MSE_matrix(c, 2:end) = [Jondrow1982_mid_quartile_u_hat_MSE, NW_u_hat_mid_quartile_u_hat_MSE, NW_u_hat_conditional_eps_lower_quartile_u_hat_MSE];
        u_hat_lower_quartile_MSE_matrix(c, 2:end) = [Jondrow1982_lower_quartile_u_hat_MSE, NW_u_hat_lower_quartile_u_hat_MSE, NW_u_hat_conditional_eps_mid_quartile_u_hat_MSE];
        Logl_matrix(c, 2:end) = [APS14_logL];
    end
    
    Logl_matrix = mean(Logl_matrix);
    u_hat_MSE_std_matrix = std(u_hat_MSE_matrix);
    u_hat_MSE_matrix = mean(u_hat_MSE_matrix); 
    u_hat_upper_quartile_MSE_matrix = mean(u_hat_upper_quartile_MSE_matrix);
    u_hat_mid_quartile_MSE_matrix = mean(u_hat_mid_quartile_MSE_matrix);
    u_hat_lower_quartile_MSE_matrix = mean(u_hat_lower_quartile_MSE_matrix);
    
    Logl_matrix(:,1) = rho1;
    u_hat_MSE_matrix(:,1) = rho1;
    u_hat_upper_quartile_MSE_matrix(:,1) = rho1;
    u_hat_mid_quartile_MSE_matrix(:,1) = rho1;
    u_hat_lower_quartile_MSE_matrix(:,1) = rho1;
    
    %Mean of all elementary scores for Murphy Diagrams
    avg_JLMS_Murphy_scores_matrix = mean(JLMS_Murphy_scores_matrix, 2);
    avg_NW_Murphy_scores_matrix = mean(NW_Murphy_scores_matrix, 2);
    avg_NW_conditional_W_Murphy_scores_matrix = mean(NW_conditional_W_Murphy_scores_matrix, 2);

        %Find a range for theta
    max_tmp = sqrt(true_sigma2_u).*norminv((0.999+1)/2, 0, 1); %set the max theta to the 99.9th quantile of the distribution of technical inefficiency
    min_tmp = 0; %theoretical minimum of the technical inefficiency predictions
    
    tmp = [min_tmp-0.1*(max_tmp - min_tmp), max_tmp + 0.1*(max_tmp - min_tmp)];
    Murphy_theta = linspace(tmp(1), tmp(2), 501)';
    
    %Export results
    Logl_matrix = array2table(Logl_matrix);
    Logl_cols_names = {'AR_1', {'Logl'}};
    Logl_matrix.Properties.VariableNames(:) = cat(2,Logl_cols_names{:});
        %MSE matrices
    u_hat_MSE_matrix = array2table(u_hat_MSE_matrix);
    u_hat_MSE_matrix.Properties.VariableNames(:) = column_labels;
    
    u_hat_upper_quartile_MSE_matrix = array2table(u_hat_upper_quartile_MSE_matrix);
    u_hat_upper_quartile_MSE_matrix.Properties.VariableNames(:) = column_labels;
    
    u_hat_mid_quartile_MSE_matrix = array2table(u_hat_mid_quartile_MSE_matrix);
    u_hat_mid_quartile_MSE_matrix.Properties.VariableNames(:) = column_labels;
    
    u_hat_lower_quartile_MSE_matrix = array2table(u_hat_lower_quartile_MSE_matrix);
    u_hat_lower_quartile_MSE_matrix.Properties.VariableNames(:) = column_labels;
    
    u_hat_MSE_std_matrix = array2table(u_hat_MSE_std_matrix);
    u_hat_MSE_std_matrix.Properties.VariableNames(:) = column_labels;
        %Elementary Score Matrices
    avg_JLMS_Murphy_scores_matrix = array2table([Murphy_theta, avg_JLMS_Murphy_scores_matrix]);
    avg_JLMS_Murphy_scores_matrix.Properties.VariableNames(:) = {'Theta', 'JLMS_elementary_score'};
    avg_NW_Murphy_scores_matrix = array2table([Murphy_theta, avg_NW_Murphy_scores_matrix]);
    avg_NW_Murphy_scores_matrix.Properties.VariableNames(:) = {'Theta', 'NW_elementary_score'};
    avg_NW_conditional_W_Murphy_scores_matrix = array2table([Murphy_theta, avg_NW_conditional_W_Murphy_scores_matrix]);
    avg_NW_conditional_W_Murphy_scores_matrix.Properties.VariableNames(:) = {'Theta', 'NW_conditional_W_elementary_score'};
    
    writetable(Logl_matrix,[filepath filesep fullfile('panel_SFA_simulation_Results', copula, sprintf('AR_1=%.2f', rho1),sprintf('Logl_AR_1=%.2f_N=%d_T=%d.csv', rho1, N, T))])
    writetable(u_hat_MSE_matrix,[filepath filesep fullfile('panel_SFA_simulation_Results', copula, sprintf('AR_1=%.2f', rho1),sprintf('u_hat_MSE_results AR_1=%.2f N=%d T=%d.csv', rho1, N, T))])
    writetable(u_hat_upper_quartile_MSE_matrix,[filepath filesep fullfile('panel_SFA_simulation_Results', copula, sprintf('AR_1=%.2f', rho1),sprintf('u_hat_upper_quartile_MSE_results AR_1=%.2f N=%d T=%d.csv', rho1, N, T))])
    writetable(u_hat_mid_quartile_MSE_matrix,[filepath filesep fullfile('panel_SFA_simulation_Results', copula, sprintf('AR_1=%.2f', rho1),sprintf('u_hat_mid_quartile_MSE_results AR_1=%.2f N=%d T=%d.csv', rho1, N, T))])
    writetable(u_hat_lower_quartile_MSE_matrix,[filepath filesep fullfile('panel_SFA_simulation_Results', copula, sprintf('AR_1=%.2f', rho1),sprintf('u_hat_lower_quartile_MSE_results AR_1=%.2f N=%d T=%d.csv', rho1, N, T))])
    
    writetable(u_hat_MSE_std_matrix,[filepath filesep fullfile('panel_SFA_simulation_Results', copula, sprintf('AR_1=%.2f', rho1),sprintf('u_hat_MSE_std_results AR_1=%.2f N=%d T=%d.csv', rho1, N, T))])

    writetable(avg_JLMS_Murphy_scores_matrix,[filepath filesep fullfile('panel_SFA_simulation_Results', copula, sprintf('AR_1=%.2f', rho1),sprintf('JLMS_Murphy_Diagram_results_SFA_panel_simulation AR_1=%.2f N=%d T=%d.csv', rho1, N, T))])
    writetable(avg_NW_Murphy_scores_matrix,[filepath filesep fullfile('panel_SFA_simulation_Results', copula, sprintf('AR_1=%.2f', rho1),sprintf('NW_Murphy_Diagram_results_SFA_simulation AR_1=%.2f N=%d T=%d.csv', rho1, N, T))])
    writetable(avg_NW_conditional_W_Murphy_scores_matrix,[filepath filesep fullfile('panel_SFA_simulation_Results', copula, sprintf('AR_1=%.2f', rho1),sprintf('NW_conditional_eps_Murphy_Diagram_results_SFA_simulation AR_1=%.2f N=%d T=%d.csv', rho1, N, T))])       

    %zip the simulation data directory and remove the original directory
    zip([filepath filesep fullfile('Datasets','panel_simulation_data',sprintf('N=%d_T=%d',N, T),sprintf('%s.zip',export_label))],...
        [filepath filesep fullfile('Datasets','panel_simulation_data',sprintf('N=%d_T=%d',N, T),sprintf('%s',export_label))]);
    rmdir([filepath filesep fullfile('Datasets','panel_simulation_data',sprintf('N=%d_T=%d',N, T),sprintf('%s',export_label))], 's')
end
    
    