function cross_sectional_SFA_RS2007_US_electricty()
    filepath = 'C:\Users\Robert James\Dropbox (Sydney Uni)\Estimating Technical Inefficiency Project';
    addpath([filepath filesep fullfile('Code', 'cross_sectional_SFA_simulation')])
    addpath([filepath filesep fullfile('Code')])
    
    if ~exist(convertCharsToStrings([filepath filesep fullfile('cross_sectional_SFA_application_Results')]))
        mkdir(convertCharsToStrings([filepath filesep fullfile('cross_sectional_SFA_application_Results')]));
    end
    
    data = readtable([filepath filesep fullfile('Datasets', 'Panel', 'RS2007_data.xlsx')]);
    data = table2array(data);
    
    fuel_price_index = [data(1:end-1,3)./data(2:end,3); 0];
    LM_price_index = [(data(1:end-1,4))./(data(2:end,4)); 0];
    data = [data, fuel_price_index, LM_price_index]; 
    data(14:14:1008, : ) = [];
    data(isnan(data(:,4)), :) = []; % plant 10 year 98 gives NAN for fuel index
    data(isnan(data(:,17)), :) = []; % plan 10 year 97 gives NAN for LM index
    %data = sortrows(data, [2,1]); %sort data by year then firm 
    
    P1 = data(:,16); % fuel price index
    P2 = data(:,17); % L&M price index
    P3 = data(:,5); % user cost of capital
    P = [P1, P2, P3];
    
    X1 = data(:, 10)./P1*1000*1e-6; % fuel costs over price index
    X2 = data(:, 11)./P2*1000*1e-6; % LM costs over price index
    X3 = data(:,8)/1000; % capital
    X = [X1, X2, X3];
    
    y = data(:,9)*1e-6; % output ml MWpH 
    
    %log transform outputs, inputs and prices 
    P = log(P);
    X = log(X);
    y = log(y);
    
    %summary statiatics
    column_labels = {'Variable', 'Mean', 'Median', 'Std', 'Min', 'Max'};
    row_labels = {'ln_y', 'ln_x1', 'ln_x2', 'ln_x3', 'ln_p1', 'ln_p2', 'ln_p3'};
    summary_stats = zeros(length(row_labels), length(column_labels));
    variable_data = [y, X, P];
    
    summary_stats(:,2) = mean(variable_data)'; %mean
    summary_stats(:,3) = median(variable_data)'; %median
    summary_stats(:,4) = std(variable_data)'; %std
    summary_stats(:,5) = min(variable_data)'; %min
    summary_stats(:,6) = max(variable_data)'; %max
    summary_stats_table = num2cell(summary_stats);
    summary_stats_table(:,1) = row_labels;
    summary_stats_table = cell2table(summary_stats_table);
    summary_stats_table.Properties.VariableNames(:) = {'Variable', 'Mean', 'Median', 'Std', 'Min', 'Max'};
    writetable(summary_stats_table, [filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_summary_statistics.csv')]);
    
    %% Estimate models
    n = length(data);
    S = 500; %Number of Halton draws used in maximum simulated likelihood - make this number grow with sample size (n) to ensure consistency
    S_kernel = 10000; %number of simulated draws for evaluation of the conditional expectation
    n_inputs = 3;
    n_corr_terms = ((n_inputs)^2-(n_inputs))/2; %Number of off diagonal lower triangular correlation/covariance terms for Gaussian copula correlation matrix. 
    
    %Create RVs for simulated ML 
    p = sobolset(1);
    p_scramble = scramble(p,'MatousekAffineOwen');
    us_ = norminv((net(p_scramble, n)+1)/2, 0, 1);
    us_Sxn = reshape(repelem(us_', S), n, S)';
    
    %% Estimate copula based cross-sectional SFA model
    initial_lalpha = -2.6;
    initial_lbeta = log([0.5, 0.2, 0.2]);
    initial_sigma2_v = 0.015;
    initial_lsigma2_v = log(initial_sigma2_v);
    initial_sigma2_u = 0.15;
    initial_lsigma2_u = log(initial_sigma2_u);
    initial_mu_W = [0.5, 0];
    
    %QMLE
    initial_Sigma = [0.2, 0.09; 0.09, 0.2];
    initial_chol_Sigma = MatrixToCholeski(initial_Sigma);
    
    QMLE_theta0 = [initial_lalpha, initial_lbeta, initial_lsigma2_v, initial_lsigma2_u, initial_mu_W, initial_chol_Sigma];
    Options = optimset('TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'notify');
    [QMLE_theta, QMLE_Logl] = fminunc(@(theta)Loglikelihood_QMLE_cross_sectional_SFA(theta, y, X, P, n_inputs, n_corr_terms), QMLE_theta0, Options);
    QMLE_theta(1:3+n_inputs) = exp(QMLE_theta(1:3+n_inputs)); %alpha, beta, sigma2_v, sigma2_u
    QMLE_Sigma_W_tril = QMLE_theta(length(QMLE_theta)+1-n_corr_terms:end);
    QMLE_Sigma_W_hat = CholeskiToMatrix(QMLE_Sigma_W_tril, n_inputs-1); 
    
    QMLE_logL = QMLE_Logl*-1;
    QMLE_AIC = 2*length(QMLE_theta0) - 2*QMLE_logL;
    QMLE_BIC = length(QMLE_theta0)*log(n) - 2*QMLE_logL;
    
    %SL80
    initial_SL80_Sigma_uW = [0.06, 0.06];
    initial_SL80_Sigma = [initial_sigma2_u , initial_SL80_Sigma_uW; initial_SL80_Sigma_uW', initial_Sigma];
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
    
    %Gaussian Copula
    initial_sigma2_w = diag(initial_Sigma);
    initial_lsigma2_w = log(initial_sigma2_w);
    
    initial_beta = [0.5, 0.3, 0.3];
    eps_ = y - initial_lalpha - X*initial_beta';
    W_ = (reshape(repmat(X(:,1), n_inputs-1, 1),n,n_inputs-1) - X(:,2:end)) - (P(:,2:end) - reshape(repmat(P(:,1), n_inputs-1, 1),n,n_inputs-1) + (log(initial_beta(1)) - log(initial_beta(2:end))));
    initial_Rho = corrcoef([eps_, W_]);
    initial_lRho = direct_mapping_mat(initial_Rho);
    
    Gaussian_copula_theta0 = [initial_lalpha, initial_lbeta, initial_lsigma2_v, initial_lsigma2_u, initial_lsigma2_w', initial_mu_W, initial_lRho'];
    Options = optimset('TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'notify');
    [Gaussian_copula_theta, Gaussian_copula_Logl] = fminunc(@(theta)Loglikelihood_Gaussian_copula_cross_sectional_application_SFA(theta, y, X, P, us_Sxn, n_inputs, n_corr_terms, S), Gaussian_copula_theta0, Options);
    Gaussian_copula_theta(1:(4+n_inputs)+(n_inputs-2)) = exp(Gaussian_copula_theta(1:(4+n_inputs)+(n_inputs-2))); %exp transform of alpha, beta, sigma2_v, sigma2_u, sigma2_w
    rhos_log_form = Gaussian_copula_theta(length(Gaussian_copula_theta)+1-n_corr_terms:end);
    Rho = inverse_mapping_vec(rhos_log_form);
    Gaussian_copula_theta(length(Gaussian_copula_theta)+1-n_corr_terms:end) = Rho(itril(size(Rho), -1))';
    
    Gaussian_copula_logL = Gaussian_copula_Logl*-1;
    Gaussian_copula_AIC = 2*length(Gaussian_copula_theta0) - 2*Gaussian_copula_logL;
    Gaussian_copula_BIC = length(Gaussian_copula_theta0)*log(n) - 2*Gaussian_copula_logL;
    
    %export LL and BIC matrix
    model_info_table = cell2table({{'Loglikelihood'}, {QMLE_logL}, {SL80_Logl}, {Gaussian_copula_logL}; {'BIC'}, {QMLE_BIC}, {SL80_BIC}, {Gaussian_copula_BIC}});
    model_info_table.Properties.VariableNames = {'Criteria', 'QMLE', 'SL80', 'Gaussian_Copula'};
    writetable(model_info_table, [filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_model_info.csv')]);
    
    %% Technical Inefficiency estimation 
        %JLMS
    [QMLE_JLMS_u_hat, QMLE_JLMS_V_u_hat] = Estimate_Jondrow1982_u_hat(QMLE_theta', n_inputs, n_corr_terms, y, X);
    [SL80_JLMS_u_hat, SL80_JLMS_V_u_hat] = Estimate_Jondrow1982_u_hat([SL80_theta(1:2+n_inputs), SL80_Sigma(1,1)]', n_inputs, n_corr_terms, y, X);
    [Gaussian_copula_JLMS_u_hat, Gaussian_copula_JLMS_V_u_hat] = Estimate_Jondrow1982_u_hat(Gaussian_copula_theta', n_inputs, n_corr_terms, y, X);
    JLMS_u_hat_matrix = [data(:,2), QMLE_JLMS_u_hat, SL80_JLMS_u_hat, Gaussian_copula_JLMS_u_hat];
    JLMS_V_u_hat_matrix = [data(:,2), QMLE_JLMS_V_u_hat, SL80_JLMS_V_u_hat, Gaussian_copula_JLMS_V_u_hat];
    JLMS_u_hat_year_mean = zeros(13, 4);
    JLMS_V_u_hat_year_mean = zeros(13, 4);
    JLMS_std_u_hat_year_mean = zeros(13, 4);
    i = 1;
    for t=86:98
        JLMS_u_hat_year_mean(i, 1) = t;
        JLMS_V_u_hat_year_mean(i, 1) = t;
        JLMS_std_u_hat_year_mean(i, 1) = t;
        JLMS_u_hat_year_mean(i, 2:end) = mean(JLMS_u_hat_matrix(JLMS_u_hat_matrix(:,1) == t, 2:end));
        JLMS_V_u_hat_year_mean(i, 2:end) = mean(JLMS_V_u_hat_matrix(JLMS_V_u_hat_matrix(:,1) == t, 2:end));
        JLMS_std_u_hat_year_mean(i, 2:end) = mean(sqrt(JLMS_V_u_hat_matrix(JLMS_V_u_hat_matrix(:,1) == t, 2:end)));
        i = i + 1;
    end
    JLMS_u_hat_year_mean_table = array2table(round(JLMS_u_hat_year_mean, 4));
    JLMS_V_u_hat_year_mean_table = array2table(round(JLMS_V_u_hat_year_mean, 4));
    JLMS_std_u_hat_year_mean_table = array2table(round(JLMS_std_u_hat_year_mean, 4));
    JLMS_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'QMLE', 'SL80', 'Gaussian_Copula'};
    JLMS_V_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'QMLE', 'SL80', 'Gaussian_Copula'};
    JLMS_std_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'QMLE', 'SL80', 'Gaussian_Copula'};
    writetable(JLMS_u_hat_year_mean_table, [filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_JLMS_yearly_mean_TI_scores.csv')]);
    writetable(JLMS_V_u_hat_year_mean_table, [filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_JLMS_yearly_mean_Variance_TI_scores.csv')]);
    writetable(JLMS_std_u_hat_year_mean_table, [filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_JLMS_yearly_mean_std_TI_scores.csv')]);
    
        %Copula Nadaraya Watson
    Gaussian_copula_U_hat = simulate_error_components_cross_sectional_SFA(Gaussian_copula_theta(length(Gaussian_copula_theta)+1-n_corr_terms:end), 'Gaussian', n_inputs, S_kernel, 1234);
    [Gaussian_copula_NW_conditional_W_u_hat, Gaussian_copula_NW_conditional_W_V_u_hat] = Estimate_NW_u_hat_conditional_W_cross_sectional_application(Gaussian_copula_theta', n_inputs, n_corr_terms, y, X, P, Gaussian_copula_U_hat, S_kernel);
    NW_conditional_W_u_hat_matrix = [data(:,2), Gaussian_copula_NW_conditional_W_u_hat];
    NW_conditional_W_V_u_hat_matrix = [data(:,2), Gaussian_copula_NW_conditional_W_V_u_hat];
    NW_conditional_W_u_hat_year_mean = zeros(13, 2);
    NW_conditional_W_V_u_hat_year_mean = zeros(13, 2);
    NW_conditional_W_std_u_hat_year_mean = zeros(13, 2);
    i = 1;
    for t=86:98
        NW_conditional_W_u_hat_year_mean(i, 1) = t;
        NW_conditional_W_V_u_hat_year_mean(i, 1) = t;
        NW_conditional_W_std_u_hat_year_mean(i, 1) = t;
        NW_conditional_W_u_hat_year_mean(i, 2:end) = mean(NW_conditional_W_u_hat_matrix(NW_conditional_W_u_hat_matrix(:,1) == t, 2:end));
        NW_conditional_W_V_u_hat_year_mean(i, 2:end) = mean(NW_conditional_W_V_u_hat_matrix(NW_conditional_W_V_u_hat_matrix(:,1) == t, 2:end));
        NW_conditional_W_std_u_hat_year_mean(i, 2:end) = mean(sqrt(NW_conditional_W_V_u_hat_matrix(NW_conditional_W_V_u_hat_matrix(:,1) == t, 2:end)));
        i = i + 1;
    end   
    NW_conditional_W_u_hat_year_mean_table = array2table(round(NW_conditional_W_u_hat_year_mean, 4));
    NW_conditional_W_V_u_hat_year_mean_table = array2table(round(NW_conditional_W_V_u_hat_year_mean, 4));
    NW_conditional_W_std_u_hat_year_mean_table = array2table(round(NW_conditional_W_std_u_hat_year_mean, 4));
    NW_conditional_W_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'Gaussian_Copula'};
    NW_conditional_W_V_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'Gaussian_Copula'};
    writetable(NW_conditional_W_u_hat_year_mean_table, [filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_NW_conditional_W_yearly_mean_TI_scores.csv')]);
    writetable(NW_conditional_W_V_u_hat_year_mean_table, [filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_NW_conditional_W_yearly_mean_Variance_TI_scores.csv')]);
    writetable(NW_conditional_W_std_u_hat_year_mean_table, [filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_NW_conditional_W_yearly_mean_std_TI_scores.csv')]);
        
        %Export simulated training data and compute LLF u hat
    export_simulation_data_RS2007_electricity_application(Gaussian_copula_theta', n_inputs, n_corr_terms, y, X, P, Gaussian_copula_U_hat, S_kernel, filepath)
    system('"C:\Program Files\R\R-4.1.2\bin\Rscript.exe" "C:\Users\Robert James\Dropbox (Sydney Uni)\Estimating Technical Inefficiency Project\Code\Application\cross_sectional_SFA\train_LocalLinear_forest_cross_sectional_RS2007_electricty_application.R"')
    Gaussian_copula_LLF_results = table2array(readtable([filepath filesep fullfile('cross_sectional_SFA_application_Results', 'LLF_Gaussian_copula_u_hat.csv')]));
    Gaussian_copula_LLF_conditional_W_u_hat = Gaussian_copula_LLF_results(:,1);
    Gaussian_copula_LLF_conditional_W_V_u_hat = Gaussian_copula_LLF_results(:,2);
    
    LLF_u_hat_matrix = [data(:,2), Gaussian_copula_LLF_conditional_W_u_hat];
    LLF_V_u_hat_matrix = [data(:,2), Gaussian_copula_LLF_conditional_W_V_u_hat];
    LLF_u_hat_year_mean = zeros(13, 2);
    LLF_V_u_hat_year_mean = zeros(13, 2);
    LLF_std_u_hat_year_mean = zeros(13, 2);
    i = 1;
    for t=86:98
        LLF_u_hat_year_mean(i, 1) = t;
        LLF_V_u_hat_year_mean(i, 1) = t;
        LLF_std_u_hat_year_mean(i, 1) = t;
        LLF_u_hat_year_mean(i, 2:end) = mean(LLF_u_hat_matrix(LLF_u_hat_matrix(:,1) == t, 2:end));
        LLF_V_u_hat_year_mean(i, 2:end) = mean(LLF_V_u_hat_matrix(LLF_V_u_hat_matrix(:,1) == t, 2:end));
        LLF_std_u_hat_year_mean(i, 2:end) = mean(sqrt(LLF_V_u_hat_matrix(LLF_V_u_hat_matrix(:,1) == t, 2:end)));
        i = i + 1;
    end  
    LLF_u_hat_year_mean_table = array2table(round(LLF_u_hat_year_mean, 4));
    LLF_V_u_hat_year_mean_table = array2table(round(LLF_V_u_hat_year_mean, 4));
    LLF_std_u_hat_year_mean_table = array2table(round(LLF_std_u_hat_year_mean, 4));
    LLF_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'Gaussian_Copula'};
    LLF_V_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'Gaussian_Copula'};
    LLF_std_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'Gaussian_Copula'};
    writetable(LLF_u_hat_year_mean_table, [filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_LLF_yearly_mean_TI_scores.csv')]);
    writetable(LLF_V_u_hat_year_mean_table, [filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_LLF_yearly_mean_Variance_TI_scores.csv')]);
    writetable(LLF_std_u_hat_year_mean_table, [filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_LLF_yearly_mean_std_TI_scores.csv')]);
    
    E_QMLE_JLMS_u_hat = mean(QMLE_JLMS_u_hat);
    E_SL80_JLMS_u_hat = mean(SL80_JLMS_u_hat);
    E_Gaussian_copula_JLMS_u_hat = mean(Gaussian_copula_JLMS_u_hat);
    E_Gaussian_copula_NW_conditional_W_u_hat = mean(Gaussian_copula_NW_conditional_W_u_hat);
    E_Gaussian_copula_LLF_conditional_W_u_hat = mean(Gaussian_copula_LLF_conditional_W_u_hat);
    
    JLMS_mean_TI_scores = [E_QMLE_JLMS_u_hat, E_SL80_JLMS_u_hat, E_Gaussian_copula_JLMS_u_hat];
    JLMS_mean_TI_scores_table = array2table(JLMS_mean_TI_scores);
    JLMS_mean_TI_scores_table.Properties.VariableNames(:) = {'QMLE', 'SL80', 'Gaussian_Copula'};
    writetable(JLMS_mean_TI_scores_table, [filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_JLMS_mean_TI_scores.csv')]);
    
    non_parametric_regression_mean_TI_scores_table = cell2table({{'NW'}, {E_Gaussian_copula_NW_conditional_W_u_hat}; {'LLF'}, {E_Gaussian_copula_LLF_conditional_W_u_hat}});
    non_parametric_regression_mean_TI_scores_table.Properties.VariableNames(:) = {'Estimator', 'Gaussian_Copula'};
    writetable(non_parametric_regression_mean_TI_scores_table, [filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_non_parametric_regression_TI_scores.csv')]);
    
    %density plots
        %Technical Efficiency
    [Gaussian_copula_JLMS_TE_kernel, Gaussian_copula_JLMS_TE_kernel_x] = ksdensity(Gaussian_copula_JLMS_u_hat);
    [Gaussian_copula_NW_TE_kernel, Gaussian_copula_NW_TE_kernel_x] = ksdensity(Gaussian_copula_NW_conditional_W_u_hat); 
    [Gaussian_copula_LLF_TE_kernel, Gaussian_copula_LLS_TE_kernel_x] = ksdensity(Gaussian_copula_LLF_conditional_W_u_hat); 
    
    figure
    set(gcf,'position',[10,10,750,750])
    plot(Gaussian_copula_JLMS_TE_kernel_x,Gaussian_copula_JLMS_TE_kernel,'-','color','k', 'LineWidth',1);
    hold on 
    plot(Gaussian_copula_NW_TE_kernel_x,Gaussian_copula_NW_TE_kernel, ':', 'color','k', 'LineWidth',1);
    hold on
    plot(Gaussian_copula_LLS_TE_kernel_x,Gaussian_copula_LLF_TE_kernel,'-*', 'color','k', 'LineWidth',1);
    title('JLMS & APS16 Estimator', 'fontsize',14);
    xlabel('$u_{i}$', 'Interpreter','latex', 'fontsize',15);
    ylabel('Kernel Density', 'Interpreter','latex', 'fontsize',14);
    legend('JLMS $E[u_{i}|\varepsilon_{i}]$','NW $E[u_{i}|\varepsilon_{i}, \omega_{i2}, \omega_{i3}]$','LLF $E[u_{i}|\varepsilon_{i}, \omega_{i2}, \omega_{i3}]$', 'Interpreter','latex', 'Location','southoutside', 'Orientation','horizontal', 'fontsize',16, 'NumColumns',1);
    xlim([0, 1.5])
    saveas(gcf,[filepath filesep fullfile('cross_sectional_SFA_application_Results', 'RS2007_TI_kernel_density_plot.png')]);
end


    
    