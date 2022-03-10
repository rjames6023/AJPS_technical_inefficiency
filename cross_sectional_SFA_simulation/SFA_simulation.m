function SFA_simulation(n, rho_u_w1, rho_u_w2, rho_u_w3, rho_w1_w2, rho_w1_w3, rho_w2_w3, copula_misspecification)
    %filepath = 'C:\Users\Robert James\Dropbox (Sydney Uni)\Estimating Technical Inefficiency Project';
    filepath = [filesep fullfile('project','RDS-FOB-Eff_Dens_1-RW','technical_inefficiency_estimation')]; %works on windows and linux
    addpath([filepath filesep fullfile('Code')])
    
    %Simulation parameters
    num_iterations = 500;
    S = 500; %Number of Halton draws used in maximum simulated likelihood - make this number grow with sample size (n) to ensure consistency
    S_kernel = 10000; %number of simulated draws for evaluation of the conditional expectation

    if (~strcmp(rho_u_w1, '~')) && (strcmp(rho_u_w2, '~'))
       n_inputs = 2;
       rho_u_W = str2double(rho_u_w1);
       rho_W = [];
       true_beta = repelem(0.45, n_inputs);
       column_labels = {'rho_u_w1', 'JLMS','NW', 'NW_conditional_w1w2'};
       export_label = sprintf('rho=%.3f',rho_u_W);
    elseif (~strcmp(rho_u_w1, '~')) && (~strcmp(rho_u_w2, '~')) && (strcmp(rho_u_w3, '~'))
       n_inputs = 3;
       rho_u_W = [str2double(rho_u_w1), str2double(rho_u_w2)];
       rho_W = str2double(rho_w1_w2);
       true_beta = repelem(0.3, n_inputs);
       column_labels = {'rho_u_w1', 'rho_u_w2', 'rho_w1_w2', 'JLMS','NW', 'NW_conditional_w1w2'};
       export_label = sprintf('rho=%.3f %.3f',rho_u_W(1), rho_u_W(2));
    elseif(~strcmp(rho_u_w1, '~')) && (~strcmp(rho_u_w2, '~')) && (~strcmp(rho_u_w3, '~'))
       n_inputs = 4;
       rho_u_W = [str2double(rho_u_w1), str2double(rho_u_w2), str2double(rho_u_w3)];
       rho_W = [str2double(rho_w1_w2), str2double(rho_w1_w3), str2double(rho_w2_w3)];
       true_beta = repelem(0.225, n_inputs);
       column_labels = {'rho_u_w1', 'rho_u_w2', 'rho_u_w3' 'rho_w1_w2', 'rho_w1_w3', 'rho_w2_w3' 'JLMS','NW', 'NW_conditional_w1w2'};
       export_label = sprintf('rho=%.3f %.3f %.3f',rho_u_W(1), rho_u_W(2), rho_u_W(3));
    end

    n_corr_terms = ((n_inputs)^2-(n_inputs))/2; %Number of off diagonal lower triangular correlation/covariance terms for Gaussian copula correlation matrix. 

    rng(1234) %random seed
    true_sigma2_u = 1;
    true_sigma2_v = 0.5;
    true_sigma2_w = repelem(1, n_inputs-1); %variance for technical inefficiency terms
    true_sigma_P = repelem(1, n_inputs);
    true_alpha = 1; %Assumed to be on standard scale (in log scale should be 0)
    y = normrnd(0, 1, n, 1);
    rho_P = repelem(0.5,n_corr_terms);
    Rho_P = build_corr_matrix(n_inputs, rho_P);
    Sigma_P = corr2cov(sqrt(true_sigma_P), Rho_P);
    P = mvnrnd(repelem(0, n_inputs), Sigma_P, n); %assume prices are already on log scale
    
    r = sum(true_beta); %returns to scale
    X_prod_term = sum(repmat((true_beta./r),n,1).*P, 2);
    k = zeros(1, n_inputs);
    X_ = zeros(n, n_inputs);
    for i=1:n_inputs
        k(i) = true_beta(i)*(true_alpha*prod(true_beta.^true_beta))^(-1/r);
        X_(:,i) = log(k(i)) + y./r + X_prod_term - P(:,i);
    end
    
    if strcmp(copula_misspecification, '~')
        if ~exist(convertCharsToStrings([filepath filesep fullfile('Results',sprintf('n_inputs=%d', n_inputs))]))
            mkdir(convertCharsToStrings([filepath filesep fullfile('Results',sprintf('n_inputs=%d', n_inputs))]));
        end
    else
        if ~exist(convertCharsToStrings([filepath filesep fullfile(sprintf('%s_Results', copula_misspecification),sprintf('n_inputs=%d', n_inputs))]))
            mkdir(convertCharsToStrings([filepath filesep fullfile(sprintf('%s_Results', copula_misspecification),sprintf('n_inputs=%d', n_inputs))]));
        end
    end
    
    %%
    %Correlation matrix
    Rho = build_corr_matrix_u_w(n_inputs, rho_u_W, rho_W);
    Rho_eigenvalues = eig(Rho);   
    if any(Rho_eigenvalues(Rho_eigenvalues < 0)) == 1 %Check if any eigenvalue is negative, therefore Rho is not PSD
        Rho = nearcorr(Rho); %find the nearest PSD correlation matrix - via minimizing the frobenius norm
    end

    %True parameter vector. Use the nearest correlation matrix params if necessary
    idx = 1:n_inputs; 
    tril_idx = rot90(bsxfun(@plus,idx,idx(:)))-1;
    Rho_tril = Rho(tril_idx<=n_inputs+(-1)); %first n_inputs-1 entries are corr_u_W terms the rest are corr_W terms. In ascending order
    true_theta = [true_alpha, true_beta, Rho_tril', true_sigma2_v, true_sigma2_u, true_sigma2_w]';

    %Matrices to hold MSE results.
    Logl_matrix = zeros(num_iterations, n_corr_terms+1); %Column is Log-likelihood
    %Columns are: Jondrow1982, NW_u_hat, NW_u_hat_conditional_w1w2 JLMS APS16 analytic
    u_hat_upper_quartile_MSE_matrix = zeros(num_iterations, n_corr_terms+3);
    u_hat_mid_quartile_MSE_matrix = zeros(num_iterations, n_corr_terms+3);
    u_hat_lower_quartile_MSE_matrix = zeros(num_iterations, n_corr_terms+3);
    u_hat_MSE_matrix = zeros(num_iterations, n_corr_terms+3);
    std_u_hat_matrix = zeros(num_iterations, n_corr_terms+3); 
    
    %Matrices to hold Murphy Diagram score results.   
    JLMS_Murphy_scores_matrix = zeros(501, num_iterations);
    NW_Murphy_scores_matrix = zeros(501, num_iterations);
    NW_conditional_W_Murphy_scores_matrix = zeros(501, num_iterations);

    for c=1:num_iterations
        %%
        seed = c;
        rng(seed)

        %APS simulation setup with copula based dependence
        [X, u, U] = APS_simulation_setup(n, n_inputs, Rho, true_beta, X_, r, true_sigma2_u, true_sigma2_v, true_sigma2_w, seed);

        %Create RVs for simulated ML 
        p = sobolset(1);
        p_scramble = scramble(p,'MatousekAffineOwen');
        us_ = norminv((net(p_scramble, n)+1)/2, 0, 1);
        us_Sxn = reshape(repelem(us_', S), n, S)';

        %% Estimate copula based cross-sectional SFA model
            %Define initial values for maximization of the log-likelihood
        initial_alpha = true_alpha;
        initial_lalpha = log(initial_alpha);
        initial_beta = repelem(0.9/n_inputs, n_inputs);
        initial_logit_beta = log(initial_beta./(1-initial_beta));
        initial_sigma2_u = true_sigma2_u;
        initial_sigma2_v = true_sigma2_v;
        initial_lsigma2_u = log(initial_sigma2_u);
        initial_lsigma2_v = log(initial_sigma2_v);
        initial_sigma2_w = true_sigma2_w;
        initial_lsigma2_w = log(initial_sigma2_w);
        
        if strcmp(copula_misspecification, '~') %estimate the SFA model using the true dependence structure
            %Take initial values for the correlation matrix as the true correlations
            initial_rho_log_transform = direct_mapping_mat(Rho); %first n_inputs-1 entries are corr_u_W terms the rest are corr_W terms. In ascending order. log parametrization of "A New Parametrization of Correlation Matrices" 
            theta0 = [initial_lalpha, initial_logit_beta, initial_lsigma2_v, initial_lsigma2_u, initial_lsigma2_w, initial_rho_log_transform']'; 
            [theta, Logl] = Estimate_Gaussian_copula_SFA_model(theta0, n_inputs, n_corr_terms, y, X, P, us_Sxn);
        
        elseif strcmp(copula_misspecification, 'Clayton') || strcmp(copula_misspecification, 'Gumbel') || strcmp(copula_misspecification, 'Frank') %Bivariate Archimedian family
            theta0 = [initial_lalpha, initial_logit_beta, initial_lsigma2_v, initial_lsigma2_u, initial_lsigma2_w];
            [theta, Logl] = Estimate_Archimedean_vine_copula_SFA_model(y, X, P, Rho, theta0, us_Sxn, n_inputs, n_corr_terms, copula_misspecification);
        end
        
        %%
        %Simulated dependent U based upon estimated copula parameters
        if strcmp(copula_misspecification, '~') %Simulate dependent data using the "true" copula
            U_hat = simulate_error_components_cross_sectional_SFA(theta(length(theta)+1-n_corr_terms:end), 'Gaussian', n_inputs, S_kernel, seed);
        elseif strcmp(copula_misspecification, 'Clayton') || strcmp(copula_misspecification, 'Gumbel') || strcmp(copula_misspecification, 'Frank') %Bivariate Archimedian family
            U_hat = simulate_error_components_cross_sectional_SFA(theta(length(theta)+1-n_corr_terms:end), copula_misspecification, n_inputs, S_kernel, seed);
        end
        
        %Estimate the technical ineffieincy based upon ML solution
        [JLMS_u_hat, JLMS_V_u_hat] = Estimate_Jondrow1982_u_hat(theta, n_inputs, n_corr_terms, y, X);
        %Technical inefficiency using information from the joint distribution
        [NW_u_hat, NW_V_u_hat] = Estimate_NW_u_hat(theta, n_inputs, n_corr_terms, y, X, U_hat, S_kernel); %Multivariate Nadaraya Watson non-parametric estimator conditional on epsilon
        [NW_u_hat_conditional_W, NW_V_u_hat_conditional_W] = Estimate_NW_u_hat_conditional_W(theta, n_inputs, n_corr_terms, y, X, P, U_hat, S_kernel); %Multivariate Nadaraya Watson non-parametric estimator conditional on epsilon, W

        %Scores for the Murphy Diagrams
        [JLMS_S1_avg, NW_S2_avg] = Murphy_Diagram_scores(u, JLMS_u_hat, NW_u_hat, true_sigma2_u);
        [~, NW_conditional_W_S2_avg] = Murphy_Diagram_scores(u, JLMS_u_hat, NW_u_hat_conditional_W, true_sigma2_u);
        
        JLMS_Murphy_scores_matrix(:,c) = JLMS_S1_avg;
        NW_Murphy_scores_matrix(:,c) = NW_S2_avg;
        NW_conditional_W_Murphy_scores_matrix(:,c) = NW_conditional_W_S2_avg;
        
        %Export simulated datasets for python models
        export_simulation_data(theta, n_inputs, n_corr_terms, u, y, X, P, U_hat, S_kernel, c, filepath, export_label);

        %%
        %Compute MSE of technical ineffieincy estimates
        Jondrow1982_u_hat_MSE = mean((u - JLMS_u_hat).^2);
        NW_u_hat_MSE = mean((u - NW_u_hat).^2);
        NW_u_hat_conditional_w1w2_MSE = mean((u - NW_u_hat_conditional_W).^2);

         idx_upper = find(u >= quantile(u, 0.75));
         Jondrow1982_upper_quartile_u_hat_MSE = mean((u(idx_upper) - JLMS_u_hat(idx_upper)).^2);
         NW_u_hat_upper_quartile_u_hat_MSE = mean((u(idx_upper) - NW_u_hat(idx_upper)).^2);
         NW_u_hat_conditional_w1w2_upper_quartile_u_hat_MSE = mean((u(idx_upper) - NW_u_hat_conditional_W(idx_upper)).^2);

         idx_lower = find(u <= quantile(u, 0.25));
         Jondrow1982_lower_quartile_u_hat_MSE = mean((u(idx_lower) - JLMS_u_hat(idx_lower)).^2);
         NW_u_hat_lower_quartile_u_hat_MSE = mean((u(idx_lower) - NW_u_hat(idx_lower)).^2);
         NW_u_hat_conditional_w1w2_mid_quartile_u_hat_MSE = mean((u(idx_lower) - NW_u_hat_conditional_W(idx_lower)).^2);

         idx_mid = find((u > quantile(u, 0.25)) & (u < quantile(u, 0.75)));
         Jondrow1982_mid_quartile_u_hat_MSE = mean((u(idx_mid) - JLMS_u_hat(idx_mid)).^2);
         NW_u_hat_mid_quartile_u_hat_MSE = mean((u(idx_mid) - NW_u_hat(idx_mid)).^2);
         NW_u_hat_conditional_w1w2_lower_quartile_u_hat_MSE = mean((u(idx_mid) - NW_u_hat_conditional_W(idx_mid)).^2);

         
        u_hat_MSE_matrix(c, n_corr_terms+1:end) = [Jondrow1982_u_hat_MSE, NW_u_hat_MSE, NW_u_hat_conditional_w1w2_MSE];
        u_hat_upper_quartile_MSE_matrix(c, n_corr_terms+1:end) = [Jondrow1982_upper_quartile_u_hat_MSE, NW_u_hat_upper_quartile_u_hat_MSE, NW_u_hat_conditional_w1w2_upper_quartile_u_hat_MSE];
        u_hat_mid_quartile_MSE_matrix(c, n_corr_terms+1:end) = [Jondrow1982_mid_quartile_u_hat_MSE, NW_u_hat_mid_quartile_u_hat_MSE, NW_u_hat_conditional_w1w2_mid_quartile_u_hat_MSE];
        u_hat_lower_quartile_MSE_matrix(c, n_corr_terms+1:end) = [Jondrow1982_lower_quartile_u_hat_MSE, NW_u_hat_lower_quartile_u_hat_MSE, NW_u_hat_conditional_w1w2_lower_quartile_u_hat_MSE];
        std_u_hat_matrix(c, n_corr_terms+1:end) = [mean(sqrt(JLMS_V_u_hat)), mean(sqrt(NW_V_u_hat)), mean(sqrt(NW_V_u_hat_conditional_W))];
        Logl_matrix(c, n_corr_terms+1) = Logl;
        
    end
    
    u_hat_MSE_matrix_export = array2table(u_hat_MSE_matrix);
    u_hat_MSE_matrix_export.Properties.VariableNames(:) = column_labels;
    writetable(u_hat_MSE_matrix_export,[filepath filesep fullfile('Results',sprintf('n_inputs=%d', n_inputs),sprintf('u_hat_MSE_individual_replication_results %s n_inputs=%d n=%d.csv', export_label, n_inputs, n))])
    
    Logl_matrix = mean(Logl_matrix);
    u_hat_MSE_std_matrix = std(u_hat_MSE_matrix);
    u_hat_MSE_matrix = mean(u_hat_MSE_matrix);
    u_hat_upper_quartile_MSE_matrix = mean(u_hat_upper_quartile_MSE_matrix);
    u_hat_mid_quartile_MSE_matrix = mean(u_hat_mid_quartile_MSE_matrix);
    u_hat_lower_quartile_MSE_matrix = mean(u_hat_lower_quartile_MSE_matrix);
    
    Logl_matrix(:,1:n_corr_terms) = [rho_u_W, rho_W];
    u_hat_MSE_std_matrix(:,1:n_corr_terms) = [rho_u_W, rho_W];
    u_hat_MSE_matrix(:,1:n_corr_terms) = [rho_u_W, rho_W];
    
    u_hat_upper_quartile_MSE_matrix(:,1:n_corr_terms) = [rho_u_W, rho_W];
    u_hat_mid_quartile_MSE_matrix(:,1:n_corr_terms) = [rho_u_W, rho_W];
    u_hat_lower_quartile_MSE_matrix(:,1:n_corr_terms) = [rho_u_W, rho_W];
    
    %Mean of all elementary scores for Murphy Diagrams
    avg_JLMS_Murphy_scores_matrix = mean(JLMS_Murphy_scores_matrix, 2);
    avg_NW_Murphy_scores_matrix = mean(NW_Murphy_scores_matrix, 2);
    avg_NW_conditional_W_Murphy_scores_matrix = mean(NW_conditional_W_Murphy_scores_matrix, 2);
    if ~exist(convertCharsToStrings([filepath filesep fullfile('Figures','Murphy Diagrams')]))
        mkdir(convertCharsToStrings([filepath filesep fullfile('Figures','Murphy Diagrams')]));
    end
            %Find a range for theta
    max_tmp = sqrt(true_sigma2_u).*norminv((0.999+1)/2, 0, 1); %set the max theta to the 99.9th quantile of the distribution of technical inefficiency
    min_tmp = 0; %theoretical minimum of the technical inefficiency predictions

    tmp = [min_tmp-0.1*(max_tmp - min_tmp), max_tmp + 0.1*(max_tmp - min_tmp)];
    Murphy_theta = linspace(tmp(1), tmp(2), 501)';

    %Export results
    Logl_matrix = array2table(Logl_matrix);
    Logl_cols_names = {column_labels(1:n_corr_terms), {'Logl'}};
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
    
    writetable(Logl_matrix,[filepath filesep fullfile('Results',sprintf('n_inputs=%d', n_inputs),sprintf('Logl %s n_inputs=%d n=%d.csv', export_label, n_inputs, n))])
    writetable(u_hat_MSE_matrix,[filepath filesep fullfile('Results',sprintf('n_inputs=%d', n_inputs),sprintf('u_hat_MSE_results %s n_inputs=%d n=%d.csv', export_label, n_inputs, n))])
    writetable(u_hat_upper_quartile_MSE_matrix,[filepath filesep fullfile('Results',sprintf('n_inputs=%d', n_inputs),sprintf('u_hat_upper_quartile_MSE_results %s n_inputs=%d n=%d.csv', export_label, n_inputs, n))])
    writetable(u_hat_mid_quartile_MSE_matrix,[filepath filesep fullfile('Results',sprintf('n_inputs=%d', n_inputs),sprintf('u_hat_mid_quartile_MSE_results %s n_inputs=%d n=%d.csv', export_label, n_inputs, n))])
    writetable(u_hat_lower_quartile_MSE_matrix,[filepath filesep fullfile('Results',sprintf('n_inputs=%d', n_inputs),sprintf('u_hat_lower_quartile_MSE_results %s n_inputs=%d n=%d.csv', export_label, n_inputs, n))])
    
    writetable(avg_JLMS_Murphy_scores_matrix,[filepath filesep fullfile('Results',sprintf('n_inputs=%d', n_inputs),sprintf('Murphy_Diagram_results %s n_inputs=%d n=%d.csv', export_label, n_inputs, n))])
    writetable(avg_NW_Murphy_scores_matrix,[filepath filesep fullfile('Results',sprintf('n_inputs=%d', n_inputs),sprintf('Murphy_Diagram_results %s n_inputs=%d n=%d.csv', export_label, n_inputs, n))])
    writetable(avg_NW_conditional_W_Murphy_scores_matrix,[filepath filesep fullfile('Results',sprintf('n_inputs=%d', n_inputs),sprintf('Murphy_Diagram_results %s n_inputs=%d n=%d.csv', export_label, n_inputs, n))])
    
    fclose('all');
    %zip the simulation data directory and remove the original directory
    zip([filepath filesep fullfile('Datasets','simulation_data',sprintf('n_inputs=%d', n_inputs),sprintf('n=%d',n),sprintf('%s.zip',export_label))],...
        [filepath filesep fullfile('Datasets','simulation_data',sprintf('n_inputs=%d', n_inputs),sprintf('n=%d',n),sprintf('%s',export_label))]);
    rmdir([filepath filesep fullfile('Datasets','simulation_data',sprintf('n_inputs=%d', n_inputs),sprintf('n=%d',n),sprintf('%s',export_label))], 's')
end