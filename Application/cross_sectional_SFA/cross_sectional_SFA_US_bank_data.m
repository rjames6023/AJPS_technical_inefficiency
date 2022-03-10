function cross_sectional_SFA_US_bank_data
    filepath = 'C:\Users\Robert James\Dropbox (Sydney Uni)\Estimating Technical Inefficiency Project';
    addpath([filepath filesep fullfile('Code', 'cross_sectional_SFA_simulation')])
    addpath([filepath filesep fullfile('Code')])
    
    if ~exist(convertCharsToStrings([filepath filesep fullfile('cross_sectional_SFA_application_Results')]))
        mkdir(convertCharsToStrings([filepath filesep fullfile('cross_sectional_SFA_application_Results')]));
    end
    
    data = readtable([filepath filesep fullfile('Datasets', 'Cross Sectional', 'mkt2015-data.csv')]);
    data = data((data.year(:) == 2004), :); %just use a single cross-section  | (data.year(:) == 2005)
    data = data(:,{'y1', 'x1', 'x2', 'x3', 'x4', 'w1', 'w2', 'w3', 'w4'});
    data{:, {'y1', 'x1', 'x2', 'x3', 'x4'}} = data{:, {'y1', 'x1', 'x2', 'x3', 'x4'}};
   
    %summary statiatics
    column_labels = {'Variable', 'Mean', 'Median', 'Std', 'Min', 'Max'};
    row_labels = {'ln_consumer_loans', 'ln_labor', 'ln_capital', 'ln_funds', 'ln_interest_bearing_transactions', 'ln_price_labor', 'ln_price_capital', 'ln_price_funds', 'ln_price_interest_bearing_transactions'};
    summary_stats = zeros(length(row_labels), length(column_labels));
    variable_data = table2array(data);
    
    summary_stats(:,2) = mean(variable_data)'; %mean
    summary_stats(:,3) = median(variable_data)'; %median
    summary_stats(:,4) = std(variable_data)'; %std
    summary_stats(:,5) = min(variable_data)'; %min
    summary_stats(:,6) = max(variable_data)'; %max
    summary_stats_table = num2cell(summary_stats);
    summary_stats_table(:,1) = row_labels;
    summary_stats_table = cell2table(summary_stats_table);
    summary_stats_table.Properties.VariableNames(:) = {'Variable', 'Mean', 'Median', 'Std', 'Min', 'Max'};
    writetable(summary_stats_table, [filepath filesep fullfile('cross_sectional_SFA_application_Results', 'US_bank_data_summary_statistics.csv')]);
    
    %log inputs and ouputs
    data{:, {'y1', 'x1', 'x2', 'x3', 'x4'}} = (data{:, {'y1', 'x1', 'x2', 'x3', 'x4'}}.*1000)./10000000; %in 10 millions of USD
    data = table2array(data);
    data = log(data);
    
    %% Estimate models
    n = length(data);
    S = 500; %Number of Halton draws used in maximum simulated likelihood - make this number grow with sample size (n) to ensure consistency
    S_kernel = 10000; %number of simulated draws for evaluation of the conditional expectation
    n_inputs = 4;
    n_corr_terms = ((n_inputs)^2-(n_inputs))/2; %Number of off diagonal lower triangular correlation/covariance terms for Gaussian copula correlation matrix. 
    
    %Create RVs for simulated ML 
    p = sobolset(1);
    p_scramble = scramble(p,'MatousekAffineOwen');
    us_ = norminv((net(p_scramble, n)+1)/2, 0, 1);
    us_Sxn = reshape(repelem(us_', S), n, S)';
    
    %% Estimate copula based cross-sectional SFA model
    y = data(:,1);
    X = data(:,2:5);
    P = data(:,6:end);

    %QMLE
    initial_lalpha = 4;
    initial_beta = [0.6, 0.17, 0.2, 0.01];
    initial_lbeta = log(initial_beta);
    initial_sigma2_v = 1;
    initial_lsigma2_v = log(initial_sigma2_v);
    initial_sigma2_u = 2;
    initial_lsigma2_u = log(initial_sigma2_u);
    W_ = (reshape(repmat(X(:,1), n_inputs-1, 1),n,n_inputs-1) - X(:,2:end)) - (P(:,2:end) - reshape(repmat(P(:,1), n_inputs-1, 1),n,n_inputs-1) + (log(initial_beta(1)) - log(initial_beta(2:end))));
    initial_mu_W = mean(W_);
    initial_Sigma = cov(W_);
    initial_chol_Sigma = MatrixToCholeski(initial_Sigma);
    
    QMLE_theta0 = [initial_lalpha, initial_lbeta, initial_lsigma2_v, initial_lsigma2_u, initial_mu_W, initial_chol_Sigma];
    Options = optimset('TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'notify');
    [QMLE_theta, QMLE_Logl] = fminunc(@(theta)Loglikelihood_QMLE_cross_sectional_SFA(theta, y, X, P, n_inputs, n_corr_terms), QMLE_theta0, Options);
    QMLE_theta(1:3+n_inputs) = exp(QMLE_theta(1:3+n_inputs)); %alpha
    QMLE_Sigma_W_tril = QMLE_theta(length(QMLE_theta)+1-n_corr_terms:end);
    QMLE_Sigma_W_hat = CholeskiToMatrix(QMLE_Sigma_W_tril, n_inputs-1); 
    
    QMLE_logL = QMLE_Logl*-1;
    QMLE_AIC = 2*length(QMLE_theta0) - 2*QMLE_logL;
    QMLE_BIC = length(QMLE_theta0)*log(n) - 2*QMLE_logL;
    
    %Gaussian Copula
    initial_sigma2_w = diag(initial_Sigma);
    initial_lsigma2_w = log(initial_sigma2_w);
    eps_ = y - initial_lalpha - X*initial_beta';
    initial_Rho = corr([eps_, W_], 'Type','Kendall');
    initial_lRho = direct_mapping_mat(initial_Rho);
    Gaussian_copula_theta0 = [initial_lalpha, initial_lbeta, initial_lsigma2_v, initial_lsigma2_u, initial_lsigma2_w', initial_mu_W, initial_lRho'];
    
    Options = optimset('TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'notify');
    [Gaussian_copula_theta, Gaussian_copula_Logl] = fminunc(@(theta)Loglikelihood_Gaussian_copula_cross_sectional_application_SFA(theta, y, X, P, us_Sxn, n_inputs, n_corr_terms, S), Gaussian_copula_theta0, Options);
    
    Gaussian_copula_theta(1:(4+n_inputs)+(n_inputs-2)) = exp(Gaussian_copula_theta(1:(4+n_inputs)+(n_inputs-2))); %alpha, beta, sigma2_v, sigma2_u, sigma2_W
    rhos_log_form = Gaussian_copula_theta(length(Gaussian_copula_theta)+1-n_corr_terms:end);
    Rho = inverse_mapping_vec(rhos_log_form);
    Gaussian_copula_theta(length(Gaussian_copula_theta)+1-n_corr_terms:end) = Rho(itril(size(Rho), -1))';
    
    Gaussian_copula_logL = Gaussian_copula_Logl*-1;
    Gaussian_copula_AIC = 2*length(Gaussian_copula_theta0) - 2*Gaussian_copula_logL;
    Gaussian_copula_BIC = length(Gaussian_copula_theta0)*log(n) - 2*Gaussian_copula_logL;
    
    %SL80
    initial_SL80_Sigma = cov([eps_, W_]);
    initial_SL80_Sigma_chol = MatrixToCholeski(initial_SL80_Sigma);
    SL80_theta0 = [initial_lalpha, initial_lbeta, initial_lsigma2_v, initial_mu_W, initial_SL80_Sigma_chol];
    Options = optimset('TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'notify');
    [SL80_theta, SL80_Logl] = fminunc(@(theta)Loglikelihood_SL80_cross_sectional_SFA(theta, y, X, P, n_inputs, n_corr_terms), SL80_theta0, Options);
    SL80_theta(1:2+n_inputs) = exp(SL80_theta(1:2+n_inputs));
    SL80_Sigma_tril = SL80_theta(length(SL80_theta)+1-(length(itril(n_inputs))):end);
    SL80_Sigma = CholeskiToMatrix(SL80_Sigma_tril, n_inputs); 
    
    SL80_Logl = SL80_Logl*-1;
    SL80_AIC = 2*length(SL80_theta0) - 2*SL80_Logl;
    SL80_BIC = length(SL80_theta0)*log(n) - 2*SL80_Logl;
    
    
    %% Technical Inefficiency estimation 
        %JLMS
    [QMLE_JLMS_u_hat, QMLE_JLMS_V_u_hat] = Estimate_Jondrow1982_u_hat(QMLE_theta', n_inputs, n_corr_terms, y, X);
    [SL80_JLMS_u_hat, SL80_JLMS_V_u_hat] = Estimate_Jondrow1982_u_hat([SL80_theta(1:2+n_inputs), SL80_Sigma(1,1)]', n_inputs, n_corr_terms, y, X);
    [Gaussian_copula_JLMS_u_hat, Gaussian_copula_JLMS_V_u_hat] = Estimate_Jondrow1982_u_hat(Gaussian_copula_theta', n_inputs, n_corr_terms, y, X);
    JLMS_u_hat_matrix = [data(:,2), QMLE_JLMS_u_hat, SL80_JLMS_u_hat, Gaussian_copula_JLMS_u_hat];
    JLMS_u_hat_year_mean = zeros(13, 4);
    i = 1;
    for t=2004:2012
        JLMS_u_hat_year_mean(i, 1) = t;
        JLMS_u_hat_year_mean(i, 2:end) = mean(JLMS_u_hat_matrix(JLMS_u_hat_matrix(:,1) == t, 2:end));
        i = i + 1;
    end
    JLMS_u_hat_year_mean_table = array2table(JLMS_u_hat_year_mean);
    JLMS_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'QMLE', 'SL80', 'Gaussian_Copula'};
    writetable(JLMS_u_hat_year_mean_table, [filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_JLMS_yearly_mean_TI_scores.csv')]);
    
            %Copula Nadaraya Watson
    Gaussian_copula_U_hat = simulate_error_components_cross_sectional_SFA(Gaussian_copula_theta(length(Gaussian_copula_theta)+1-n_corr_terms:end), 'Gaussian', n_inputs, S_kernel, 1234);
    [Gaussian_copula_NW_conditional_W_u_hat, Gaussian_copula_NW_conditional_W_V_u_hat] = Estimate_NW_u_hat_conditional_W_cross_sectional_application(Gaussian_copula_theta', n_inputs, n_corr_terms, y, X, P, Gaussian_copula_U_hat, S_kernel);
    NW_conditional_W_u_hat_matrix = [data(:,2), Gaussian_copula_NW_conditional_W_u_hat];
    NW_conditional_W_u_hat_year_mean = zeros(13, 2);
    
    E_QMLE_JLMS_u_hat = mean(QMLE_JLMS_u_hat);
    E_SL80_JLMS_u_hat = mean(SL80_JLMS_u_hat);
    E_Gaussian_copula_JLMS_u_hat = mean(Gaussian_copula_JLMS_u_hat);
    E_Gaussian_copula_NW_conditional_W_u_hat = mean(Gaussian_copula_NW_conditional_W_u_hat);
    E_Gaussian_copula_LLF_conditional_W_u_hat = mean(Gaussian_copula_LLF_conditional_W_u_hat);
    
    JLMS_mean_TI_scores = [E_QMLE_JLMS_u_hat, E_SL80_JLMS_u_hat, E_Gaussian_copula_JLMS_u_hat];
    JLMS_mean_TI_scores_table = array2table(JLMS_mean_TI_scores);
    JLMS_mean_TI_scores_table.Properties.VariableNames(:) = {'QMLE', 'SL80', 'Gaussian_Copula'};

    
            %Technical Efficiency
    [Gaussian_copula_JLMS_TE_kernel, Gaussian_copula_JLMS_TE_kernel_x] = ksdensity(Gaussian_copula_JLMS_u_hat, 'Support','positive');
    [Gaussian_copula_NW_TE_kernel, Gaussian_copula_NW_TE_kernel_x] = ksdensity(Gaussian_copula_NW_conditional_W_u_hat, 'Support','positive'); 
    [Gaussian_copula_LLF_TE_kernel, Gaussian_copula_LLS_TE_kernel_x] = ksdensity(Gaussian_copula_LLF_conditional_W_u_hat, 'Support','positive'); 
    
    
    figure
    set(gcf,'position',[10,10,600,500])
    plot(Gaussian_copula_JLMS_TE_kernel_x,Gaussian_copula_JLMS_TE_kernel,'k', Gaussian_copula_NW_TE_kernel_x,Gaussian_copula_NW_TE_kernel,'b');
    xlabel('$\exp(-\hat{u})$', 'Interpreter','latex');
    ylabel('Kernel Density', 'Interpreter','latex');
    legend('JLMS $E[u|\epsilon]$','NW $E[u|\epsilon, \omega_{1}, \omega_{2}]$','LLF $E[u|\epsilon, \omega_{1}, \omega_{2}]$', 'Interpreter','latex', 'Location','southoutside', 'Orientation','horizontal');
    xlim([0 1.1])
    saveas(gcf,[filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_TI_kernel_density_plot.png')]);
    
end