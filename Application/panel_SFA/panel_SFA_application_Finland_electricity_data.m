function panel_SFA_application_Finland_electricity_data()
    filepath = 'C:\Users\Robert James\Dropbox (Sydney Uni)\Estimating Technical Inefficiency Project';
    addpath([filepath filesep fullfile('Code', 'panel_SFA_simulation')])
    addpath([filepath filesep fullfile('Code')])
    
    if ~exist(convertCharsToStrings([filepath filesep fullfile('panel_SFA_application_Results')]))
        mkdir(convertCharsToStrings([filepath filesep fullfile('panel_SFA_application_Results')]));
    end
    
    data = readtable([filepath filesep fullfile('Datasets', 'Panel', 'finland.csv')]);
    
    %summary statiatics
    column_labels = {'Variable', 'Mean', 'Median', 'Std', 'Min', 'Max'};
    row_labels = {'ln_opex', 'ln_length', 'ln_energy', 'ln_numberuser', 'loutage'};
    summary_stats = zeros(length(row_labels), length(column_labels));
    variable_data = table2array(data(:, {'lnopex', 'lnlength', 'lnenergy', 'lnnumberuser', 'loutage'}));
    
    summary_stats(:,2) = mean(variable_data)'; %mean
    summary_stats(:,3) = median(variable_data)'; %median
    summary_stats(:,4) = std(variable_data)'; %std
    summary_stats(:,5) = min(variable_data)'; %min
    summary_stats(:,6) = max(variable_data)'; %max
    summary_stats_table = num2cell(summary_stats);
    summary_stats_table(:,1) = row_labels;
    summary_stats_table = cell2table(summary_stats_table);
    summary_stats_table.Properties.VariableNames(:) = {'Variable', 'Mean', 'Median', 'Std', 'Min', 'Max'};
    writetable(summary_stats_table, [filepath filesep fullfile('panel_SFA_application_Results', 'Finland_electricity_summary_statistics.csv')]);
        
    %% Estimate Models
    N = 73;
    T = 7;
    S = 500; %number of random draws used to evaluate the simulated likelihood (FMSLE)
    S_kernel = 10000; %number of simulated draws for evaluation of the conditional expectation
    
    y_ = data(:, {'id', 'year', 'lnopex'});
    X_ = data(:, {'id', 'year', 'lnlength', 'lnenergy', 'lnnumberuser'});
    w_ = data(:, {'id', 'year', 'loutage'});
    y = cell(T, 1);
    X  = cell(T, 1);
    w = cell(T,1);
    j = 1;
    for t=2008:2014
        y__ = y_(y_.year == t, :);
        X__ = X_(X_.year == t, :);
        w__ = w_(w_.year == t, :);
        y{j} = table2array(y__(:, 3));
        X{j} = table2array(X__(:, 3:end));
        w{j} = table2array(w__(:, 3));
        j = j + 1;
    end
    y = [y{:}];
    
    %% APS14 copula dynamic panel SFA
        %Independent uniform random variables for FMSLE - assumes a Gaussian copula
    p = sobolset(T);
    p_scramble = scramble(p,'MatousekAffineOwen');
    FMSLE_us = cell(T, 1);
    us_ = net(p_scramble, S);
    us_ = norminv(us_, zeros(S, T), ones(S, T)); %transform to standard normal 
    for t=1:T
        us_Sxn = repmat(us_(1:end, t)', N, 1)';
        FMSLE_us{t} = us_Sxn;
    end
        
    initial_alpha = 7.6;
    initial_beta = [0.27, 0.0296, 0.48];
    initial_logit_beta = log(initial_beta./(1-initial_beta));
    initial_sigma2_u = 0.17;
    initial_lsigma2_u = log(initial_sigma2_u);
    initial_sigma2_v = 0.006;
    initial_lsigma2_v = log(initial_sigma2_v);
    initial_delta = [initial_lsigma2_u, 0.45];
        %create synthetic time series with approximate rho and grab corr matrix of T lags as the initial correlation matrix 
    mdl = arima('Constant',0.5,'AR',{0.9},'Variance',initial_sigma2_u);
    Y = simulate(mdl,N*T);
    lags = lagmatrix(Y, (1:T));
    lags(any(isnan(lags), 2), :) = [];
    initial_Rho = corr(lags);
    initial_lRho = direct_mapping_mat(initial_Rho); 
    
    k = length(initial_beta) + 1; %number of regressors + constant
    theta0 = [initial_alpha, initial_logit_beta, initial_delta, initial_lsigma2_v, initial_lRho'];
    Options = optimset('TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'off');
    [APS14_theta, APS14_logL] = fminunc(@(theta)Loglikelihood_APS14_dynamic_panel_SFA_u_application(theta, y, X, N, T, k, S, FMSLE_us), theta0, Options);
    
    %transform parameters back to true range
    APS14_theta(1) = exp(APS14_theta(1));
    APS14_theta(2:k) = 1./(1+exp(-APS14_theta(2:k))); %inverse logit transform of betas
    APS14_theta(k+3) = exp(APS14_theta(k+3));
    
    APS14_logL = APS14_logL*-1;
    APS14_AIC = 2*length(theta0) - 2*APS14_logL;
    APS14_BIC = k*log(N*T) - 2*APS14_logL;
    
    %QMLE estimator (independence over all i and t)
    initial_alpha = 7.6;
    initial_beta = [0.27, 0.0296, 0.48];
    initial_logit_beta = log(initial_beta./(1-initial_beta));
    initial_sigma2_u = 0.17;
    initial_lsigma2_u = log(initial_sigma2_u);
    initial_sigma2_v = 0.006;
    initial_lsigma2_v = log(initial_sigma2_v);
    
    k = length(initial_beta) + 1; %number of regressors + constant
    QMLE_theta0 = [initial_alpha, initial_logit_beta, initial_lsigma2_u, initial_lsigma2_v];
    Options = optimset('TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'off');
    [QMLE_theta, QMLE_logL] = fminunc(@(theta)Loglikelihood_QMLE_panel_SFA(theta, y, X, N, T, k), QMLE_theta0, Options);
    
    %tranform parameters back to true range
    QMLE_theta(1) = exp(QMLE_theta(1));
    QMLE_theta(2:k) = 1./(1+exp(-QMLE_theta(2:k))); %inverse logit transform of betas
    QMLE_theta(k+1) = exp(QMLE_theta(k+1));
    QMLE_theta(k+2) = exp(QMLE_theta(k+2));
    
    QMLE_logL = QMLE_logL*-1;
    QMLE_AIC = 2*length(theta0) - 2*QMLE_logL;
    QMLE_BIC = k*log(N*T) - 2*QMLE_logL;
  
    %% Estimate technical inefficiency scores
        %Simulated dependent U based upon estimated copula parameters
    APS14_U_hat = simulate_error_components_panel_SFA(APS14_theta(k+4:end), 'Gaussian', T, S_kernel, 2);

    %Estimate the technical ineffieincy based upon ML solution
    [APS14_JLMS_u_hat, APS14_JLMS_V_u_hat] = Estimate_Jondrow1982_u_hat_panel_SFA_application2(APS14_theta(1), APS14_theta(2:k), APS14_theta(k+1:k+2), APS14_theta(k+3), y, X, T, N);
    [QMLE_JLMS_u_hat, QMLE_JLMS_Vu_hat] = Estimate_Jondrow1982_u_hat_panel_SFA_application1(QMLE_theta(1), QMLE_theta(2:k), QMLE_theta(k+1), QMLE_theta(k+2), y, X, T, N);
    
    APS14_JLMS_u_hat_year_mean = mean(APS14_JLMS_u_hat);
    QMLE_JLMS_u_hat_year_mean = mean(QMLE_JLMS_u_hat);
    APS14_JLMS_V_u_hat_year_mean = mean(APS14_JLMS_V_u_hat);
    QMLE_JLMS_V_u_hat_year_mean = mean(QMLE_JLMS_Vu_hat);    
    
    JLMS_u_hat_year_mean_table = array2table([(2008:2014)', QMLE_JLMS_u_hat_year_mean', APS14_JLMS_u_hat_year_mean']);
    JLMS_V_u_hat_year_mean_table = array2table([(2008:2014)', QMLE_JLMS_V_u_hat_year_mean', APS14_JLMS_V_u_hat_year_mean']);
    
    JLMS_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'QMLE', 'Gaussian_Copula'};
    JLMS_V_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'QMLE', 'Gaussian_Copula'};
    writetable(JLMS_u_hat_year_mean_table, [filepath filesep fullfile('panel_SFA_application_Results', 'Finland_electricity_JLMS_yearly_mean_TI_scores.csv')]);
    writetable(JLMS_u_hat_year_mean_table, [filepath filesep fullfile('panel_SFA_application_Results', 'Finland_electricity_JLMS_yearly_mean_Variance_TI_scores.csv')]);
    
    %Technical inefficiency using information from the joint distribution
    [APS14_NW_u_hat_conditional_eps, APS14_NW_V_u_hat_conditional_eps] = Estimate_NW_u_hat_conditional_eps_panel_SFA(APS14_theta, y, X, N, T, k, APS14_U_hat, S_kernel); %Multivariate Nadaraya Watson non-parametric estimator conditional on epsilons
    
    APS14_NW_u_hat_conditional_eps_year_mean = mean(APS14_NW_u_hat_conditional_eps);
    APS14_NW_V_u_hat_conditional_eps_year_mean = mean(APS14_NW_V_u_hat_conditional_eps);
    
    APS14_NW_u_hat_conditional_eps_u_hat_year_mean_table = array2table([(2008:2014)', APS14_NW_u_hat_conditional_eps_year_mean']);
    APS14_NW_V_u_hat_conditional_eps_u_hat_year_mean_table = array2table([(2008:2014)', APS14_NW_V_u_hat_conditional_eps_year_mean']);
    
    APS14_NW_u_hat_conditional_eps_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'Gaussian_Copula'};
    APS14_NW_V_u_hat_conditional_eps_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'Gaussian_Copula'};
    writetable(APS14_NW_u_hat_conditional_eps_u_hat_year_mean_table, [filepath filesep fullfile('panel_SFA_application_Results', 'Finland_electricity_NW_yearly_mean_TI_scores.csv')]);
    writetable(APS14_NW_V_u_hat_conditional_eps_u_hat_year_mean_table, [filepath filesep fullfile('panel_SFA_application_Results', 'Finland_electricity_NW_yearly_mean_Variance_TI_scores.csv')]);
    
    %export simulated data for other non parametric regression models
    export_finland_electricity_SFA_panel_data(APS14_theta, y, X, N, T, k, APS14_U_hat, S_kernel, filepath);
        %Run LLF R model
    system('"C:\Program Files\R\R-4.1.2\bin\Rscript.exe" "C:\Users\Robert James\Dropbox (Sydney Uni)\Estimating Technical Inefficiency Project\Code\Application\panel_SFA\train_LocalLinear_forest_panel_finnish_electricty_application.R"')
    APS14_LLF_conditional_eps_u_hat = table2array(readtable([filepath filesep fullfile('panel_SFA_application_Results', 'finnish_electricity_LLF_Gaussian_copula_u_hat.csv')]));
    APS14_LLF_conditional_eps_V_u_hat = table2array(readtable([filepath filesep fullfile('panel_SFA_application_Results', 'finnish_electricity_LLF_Gaussian_copula_V_u_hat.csv')]));

    APS14_LLF_conditional_eps_u_hat_year_mean = mean(APS14_LLF_conditional_eps_u_hat);
    APS14_LLF_conditional_eps_V_u_hat_year_mean = mean(APS14_LLF_conditional_eps_V_u_hat);

    LLF_u_hat_year_mean_table = array2table([(2008:2014)', APS14_LLF_conditional_eps_u_hat_year_mean']);
    LLF_V_u_hat_year_mean_table = array2table([(2008:2014)', APS14_LLF_conditional_eps_V_u_hat_year_mean']);
    LLF_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'Gaussian_Copula'};
    LLF_V_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'Gaussian_Copula'};
    writetable(LLF_u_hat_year_mean_table, [filepath filesep fullfile('panel_SFA_application_Results', 'Finland_electricity_LLF_yearly_mean_TI_scores.csv')]);
    writetable(LLF_V_u_hat_year_mean_table, [filepath filesep fullfile('panel_SFA_application_Results', 'Finland_electricity_LLF_yearly_mean_Variance_TI_scores.csv')]);
    
end