function panel_SFA_application_RS2007_US_electricty()
    filepath = 'C:\Users\Robert James\Dropbox (Sydney Uni)\Estimating Technical Inefficiency Project';
    addpath([filepath filesep fullfile('Code', 'panel_SFA_simulation')])
    addpath([filepath filesep fullfile('Code')])
    
    if ~exist(convertCharsToStrings([filepath filesep fullfile('panel_SFA_application_Results')]))
        mkdir(convertCharsToStrings([filepath filesep fullfile('panel_SFA_application_Results')]));
    end
    
    data = readtable([filepath filesep fullfile('Datasets', 'Panel', 'RS2007_data.xlsx')]);
    data = table2array(data);
    
    fuel_price_index = [data(1:end-1,3)./data(2:end,3); 0];
    LM_price_index = [(data(1:end-1,4))./(data(2:end,4)); 0];
    data = [data, fuel_price_index, LM_price_index]; 
    data(14:14:1008, : ) = [];
    data(isnan(data(:,4)), :) = []; % plant 10 year 98 gives NAN for fuel index
    data(isnan(data(:,17)), :) = []; % plant 10 year 97 gives NAN for LM index
    %data = sortrows(data, [2,1]); %sort data by year then firm 
    
    N_index = data(:,1);
    T_index = data(:,2);
    
    P1 = data(:,16); % fuel price index
    P2 = data(:,17); % L&M price index
    P3 = data(:,5); % user cost of capital
    P_ = [P1, P2, P3];
    
    X1 = data(:, 10)./P1*1000*1e-6; % fuel costs over price index
    X2 = data(:, 11)./P2*1000*1e-6; % LM costs over price index
    X3 = data(:,8)/1000; % capital
    X_ = [X1, X2, X3];
    
    y_ = data(:,9)*1e-6; % output ml MWpH 
    
    %log transform outputs, inputs and prices 
    P_ = log(P_);
    X_ = log(X_);
    y_ = log(y_);
    
    all_ = [N_index, T_index, y_, X_];
    
    %% Estimate Models
    N = 72;
    T = length(unique(T_index));
    S = 500; %number of random draws used to evaluate the simulated likelihood (FMSLE)
    S_kernel = 10000; %number of simulated draws for evaluation of the conditional expectation
    
    y = cell(T, 1);
    X  = cell(T, 1);
    j = 1;
    for t=86:98
        y__ = all_(all_(:,2) == t, 3);
        X__ = all_(all_(:,2) == t, 4:6);
        y{j} = y__;
        X{j} = X__;
        j = j + 1;
    end
    
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
    
    initial_lalpha = -2.6;
    initial_beta = [0.5, 0.2, 0.2];
    initial_logit_beta = log(initial_beta./(1-initial_beta));
    initial_sigma2_v = 0.015;
    initial_lsigma2_v = log(initial_sigma2_v);
    initial_sigma2_u = 0.15;
    initial_lsigma2_u = log(initial_sigma2_u);
    initial_delta = [initial_lsigma2_u, 0.45];
    
    eps_ = cell(T, 1);
    for t=1:T
        eps__ = zeros(N, 1);
        tmp_eps = y{t} - initial_lalpha- X{t}*initial_beta'; 
        eps__(1:length(tmp_eps), 1) = tmp_eps;
        eps_{t} = eps__;
    end
    initial_Rho = corr([eps_{:}]);
    initial_lRho = direct_mapping_mat(initial_Rho); 
    
    k = length(initial_beta) + 1; %number of regressors + constant
    theta0 = [initial_lalpha, initial_logit_beta, initial_delta, initial_lsigma2_v, initial_lRho'];
    Options = optimset('TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'off');
    [APS14_theta, APS14_logL] = fminunc(@(theta)Loglikelihood_APS14_dynamic_panel_SFA_u_RS2007(theta, y, X, N, T, k, S, FMSLE_us), theta0, Options);
    
    %transform parameters back to true range
    APS14_theta(1) = exp(APS14_theta(1));
    APS14_theta(2:k) = 1./(1+exp(-APS14_theta(2:k))); %inverse logit transform of betas
    APS14_theta(k+3) = exp(APS14_theta(k+3));
    
    APS14_logL = APS14_logL*-1;
    APS14_AIC = 2*length(theta0) - 2*APS14_logL;
    APS14_BIC = k*log(N*T) - 2*APS14_logL;
    
    %QMLE (independence over all i and t)
    k = length(initial_beta) + 1; %number of regressors + constant
    QMLE_theta0 = [initial_lalpha, initial_logit_beta, initial_lsigma2_u, initial_lsigma2_v];
    Options = optimset('TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'off');
    [QMLE_theta, QMLE_logL] = fminunc(@(theta)Loglikelihood_QMLE_panel_SFA_RS2007(theta, y, X, N, T, k), QMLE_theta0, Options);
    
    %tranform parameters back to true range
    QMLE_theta(1) = exp(QMLE_theta(1));
    QMLE_theta(2:k) = 1./(1+exp(-QMLE_theta(2:k))); %inverse logit transform of betas
    QMLE_theta(k+1) = exp(QMLE_theta(k+1));
    QMLE_theta(k+2) = exp(QMLE_theta(k+2));
    
    QMLE_logL = QMLE_logL*-1;
    QMLE_AIC = 2*length(theta0) - 2*QMLE_logL;
    QMLE_BIC = k*log(N*T) - 2*QMLE_logL;
    
        %export LL and BIC matrix
    model_info_table = cell2table({{'Loglikelihood'}, {QMLE_logL}, {APS14_logL}; {'BIC'}, {QMLE_BIC}, {APS14_BIC}});
    model_info_table.Properties.VariableNames = {'Criteria', 'QMLE', 'Gaussian_Copula'};
    writetable(model_info_table, [filepath filesep fullfile('panel_SFA_application_Results', 'RS2007_model_info.csv')]);
    
    %% Estimate technical inefficiency scores
        %Simulated dependent U based upon estimated copula parameters
    APS14_U_hat = simulate_error_components_panel_SFA(APS14_theta(k+4:end), 'Gaussian', T, S_kernel, 10);

    %Estimate the technical ineffieincy based upon ML solution
    [APS14_JLMS_u_hat, APS14_JLMS_V_u_hat] = Estimate_Jondrow1982_u_hat_panel_SFA_application_RS2007(APS14_theta(1), APS14_theta(2:k), APS14_theta(k+1:k+2), APS14_theta(k+3), y, X, T, N);
    [QMLE_JLMS_u_hat, QMLE_JLMS_V_u_hat] = Estimate_Jondrow1982_u_hat_panel_SFA_application1_RS2007(QMLE_theta(1), QMLE_theta(2:k), QMLE_theta(k+1), QMLE_theta(k+2), y, X, T, N);
    
    APS14_JLMS_u_hat_year_mean = round(nanmean(APS14_JLMS_u_hat), 4);
    QMLE_JLMS_u_hat_year_mean = round(nanmean(QMLE_JLMS_u_hat), 4);
    APS14_JLMS_V_u_hat_year_mean = round(nanmean(APS14_JLMS_V_u_hat), 4);
    QMLE_JLMS_V_u_hat_year_mean = round(nanmean(sqrt(QMLE_JLMS_V_u_hat)), 4); 
    APS14_JLMS_std_u_hat_year_mean = round(nanmean(sqrt(APS14_JLMS_V_u_hat)), 4);
    QMLE_JLMS_std_u_hat_year_mean = round(nanmean(sqrt(QMLE_JLMS_V_u_hat)), 4);  
    
    JLMS_u_hat_year_mean_table = array2table([(86:98)', QMLE_JLMS_u_hat_year_mean', APS14_JLMS_u_hat_year_mean']);
    JLMS_V_u_hat_year_mean_table = array2table([(86:98)', QMLE_JLMS_V_u_hat_year_mean', APS14_JLMS_V_u_hat_year_mean']);
    JLMS_std_u_hat_year_mean_table = array2table(round([(86:98)', QMLE_JLMS_std_u_hat_year_mean', APS14_JLMS_std_u_hat_year_mean'],4));
    
    JLMS_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'QMLE', 'Gaussian_Copula'};
    JLMS_V_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'QMLE', 'Gaussian_Copula'};
    JLMS_std_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'QMLE', 'Gaussian_Copula'};
    
    writetable(JLMS_u_hat_year_mean_table, [filepath filesep fullfile('panel_SFA_application_Results', 'RS2007_electricity_JLMS_yearly_mean_TI_scores.csv')]);
    writetable(JLMS_V_u_hat_year_mean_table, [filepath filesep fullfile('panel_SFA_application_Results', 'RS2007_electricity_JLMS_yearly_mean_Variance_TI_scores.csv')]);
    writetable(JLMS_std_u_hat_year_mean_table, [filepath filesep fullfile('panel_SFA_application_Results', 'RS2007_electricity_JLMS_yearly_mean_Std_TI_scores.csv')]);
    
    %Technical inefficiency using information from the joint distribution
    [APS14_NW_u_hat_conditional_eps, APS14_NW_V_u_hat_conditional_eps] = Estimate_NW_u_hat_conditional_eps_panel_SFA_RS2007(APS14_theta, y, X, N, T, k, APS14_U_hat, S_kernel); %Multivariate Nadaraya Watson non-parametric estimator conditional on epsilons
    
    APS14_NW_u_hat_conditional_eps_year_mean = round(nanmean(APS14_NW_u_hat_conditional_eps), 4);
    APS14_NW_V_u_hat_conditional_eps_year_mean = round(nanmean(APS14_NW_V_u_hat_conditional_eps), 4);
    APS14_NW_std_u_hat_conditional_eps_year_mean = round(nanmean(sqrt(APS14_NW_V_u_hat_conditional_eps)), 4);
    
    APS14_NW_u_hat_conditional_eps_u_hat_year_mean_table = array2table([(86:98)', APS14_NW_u_hat_conditional_eps_year_mean']);
    APS14_NW_V_u_hat_conditional_eps_u_hat_year_mean_table = array2table([(86:98)', APS14_NW_V_u_hat_conditional_eps_year_mean']);
    APS14_NW_std_u_hat_conditional_eps_u_hat_year_mean_table = array2table(round([(86:98)', APS14_NW_std_u_hat_conditional_eps_year_mean'],4));
    
    APS14_NW_u_hat_conditional_eps_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'Gaussian_Copula'};
    APS14_NW_V_u_hat_conditional_eps_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'Gaussian_Copula'};
    APS14_NW_std_u_hat_conditional_eps_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'Gaussian_Copula'};
    
    writetable(APS14_NW_u_hat_conditional_eps_u_hat_year_mean_table, [filepath filesep fullfile('panel_SFA_application_Results', 'RS2007_electricity_NW_yearly_mean_TI_scores.csv')]);
    writetable(APS14_NW_V_u_hat_conditional_eps_u_hat_year_mean_table, [filepath filesep fullfile('panel_SFA_application_Results', 'RS2007_electricity_NW_yearly_mean_Variance_TI_scores.csv')]);
    writetable(APS14_NW_std_u_hat_conditional_eps_u_hat_year_mean_table, [filepath filesep fullfile('panel_SFA_application_Results', 'RS2007_electricity_NW_yearly_mean_Std_TI_scores.csv')]);
    
    %export simulated data for other non parametric regression models
    export_RS2007_electricity_SFA_panel_data(APS14_theta, y, X, N, T, k, APS14_U_hat, S_kernel, filepath);
        %Run LLF R model
    system('"C:\Program Files\R\R-4.1.2\bin\Rscript.exe" "C:\Users\Robert James\Dropbox (Sydney Uni)\Estimating Technical Inefficiency Project\Code\Application\panel_SFA\RS2007\train_LocalLinear_forest_panel_RS2007_electricty_application.R"')
    APS14_LLF_conditional_eps_u_hat = readtable([filepath filesep fullfile('panel_SFA_application_Results', 'RS2007_electricity_LLF_Gaussian_copula_u_hat.csv')]);
    APS14_LLF_conditional_eps_V_u_hat = readtable([filepath filesep fullfile('panel_SFA_application_Results', 'RS2007_electricity_LLF_Gaussian_copula_V_u_hat.csv')]);
    APS14_LLF_conditional_eps_u_hat = str2double(APS14_LLF_conditional_eps_u_hat{1:end-2, :});
    APS14_LLF_conditional_eps_V_u_hat = str2double(APS14_LLF_conditional_eps_V_u_hat{1:end-2, :});

    APS14_LLF_conditional_eps_u_hat_year_mean = round(nanmean(APS14_LLF_conditional_eps_u_hat), 4);
    APS14_LLF_conditional_eps_V_u_hat_year_mean = round(nanmean(APS14_LLF_conditional_eps_V_u_hat), 4);
    APS14_LLF_conditional_eps_std_u_hat_year_mean = round(sqrt(nanmean(APS14_LLF_conditional_eps_V_u_hat)), 4);

    LLF_u_hat_year_mean_table = array2table([(86:98)', APS14_LLF_conditional_eps_u_hat_year_mean']);
    LLF_V_u_hat_year_mean_table = array2table([(86:98)', APS14_LLF_conditional_eps_V_u_hat_year_mean']);
    LLF_std_u_hat_year_mean_table = array2table([(86:98)', APS14_LLF_conditional_eps_std_u_hat_year_mean']);
    LLF_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'Gaussian_Copula'};
    LLF_V_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'Gaussian_Copula'};
    LLF_std_u_hat_year_mean_table.Properties.VariableNames(:) = {'year', 'Gaussian_Copula'};
    
    writetable(LLF_u_hat_year_mean_table, [filepath filesep fullfile('panel_SFA_application_Results', 'RS2007_electricity_LLF_yearly_mean_TI_scores.csv')]);
    writetable(LLF_V_u_hat_year_mean_table, [filepath filesep fullfile('panel_SFA_application_Results', 'RS2007_electricity_LLF_yearly_mean_Variance_TI_scores.csv')]);
    writetable(LLF_std_u_hat_year_mean_table, [filepath filesep fullfile('panel_SFA_application_Results', 'RS2007_electricity_LLF_yearly_mean_Std_TI_scores.csv')]);

    %Kernel density plots
        %All pooled observations
    APS14_JLMS_u_hat_flat = reshape(APS14_JLMS_u_hat, 1, [])';
    APS14_NW_u_hat_conditional_eps_flat = reshape(APS14_NW_u_hat_conditional_eps, 1, [])';
    APS14_LLF_conditional_eps_u_hat_flat = reshape(APS14_LLF_conditional_eps_u_hat, 1, [])';
    
    APS14_JLMS_u_hat_flat = APS14_JLMS_u_hat_flat(~isnan(APS14_JLMS_u_hat_flat));
    APS14_NW_u_hat_conditional_eps_flat = APS14_NW_u_hat_conditional_eps_flat(~isnan(APS14_NW_u_hat_conditional_eps_flat));
    APS14_LLF_conditional_eps_u_hat_flat = APS14_LLF_conditional_eps_u_hat_flat(~isnan(APS14_LLF_conditional_eps_u_hat_flat));
    
    [APS14_JLMS_u_hat_kernel, APS14_JLMS_u_hat_kernel_x] = ksdensity(APS14_JLMS_u_hat_flat);
    [APS14_NW_u_hat_conditional_eps_kernel, APS14_NW_u_hat_conditional_eps_kernel_x] = ksdensity(APS14_NW_u_hat_conditional_eps_flat); 
    [APS14_LLF_conditional_eps_u_hat_kernel, APS14_LLF_conditional_eps_u_hat_kernel_x] = ksdensity(APS14_LLF_conditional_eps_u_hat_flat); 
    
    figure
    set(gcf,'position',[10,10,750,750])
    plot(APS14_JLMS_u_hat_kernel_x,APS14_JLMS_u_hat_kernel,'-', 'color','k', 'LineWidth',1);
    hold on 
    plot(APS14_NW_u_hat_conditional_eps_kernel_x,APS14_NW_u_hat_conditional_eps_kernel,':', 'color','k', 'LineWidth',1);
    hold on
    plot(APS14_LLF_conditional_eps_u_hat_kernel_x,APS14_LLF_conditional_eps_u_hat_kernel,'-*', 'color','k', 'LineWidth',1);
    
    title('JLMS & APS14 Estimator', 'fontsize',14);
    xlabel('$u_{it}$', 'Interpreter','latex', 'fontsize',15);
    ylabel('Kernel Density', 'Interpreter','latex', 'fontsize',14);
    legend('JLMS $E[u_{it}|\varepsilon_{it}]$','NW $E[u_{it}|\varepsilon_{i1}, \dots, \varepsilon_{i,13}]$','LLF $E[u_{it}|\varepsilon_{i1}, \dots, \varepsilon_{i,13}]$', 'Interpreter','latex', 'Location','southoutside', 'Orientation','horizontal', 'fontsize',16, 'NumColumns',1);
    xlim([0, 1.5])
    saveas(gcf,[filepath filesep fullfile('panel_SFA_application_Results', 'RS2007_TI_kernel_density_plot.png')]);
end
    