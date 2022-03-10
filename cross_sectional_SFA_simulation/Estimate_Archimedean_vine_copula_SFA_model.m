function [theta, Logl] = Estimate_Archimedean_vine_copula_SFA_model(y, X, P, Rho, theta0, us_Sxn, n_inputs, n_corr_terms, copula)
    if n_inputs == 2
        %Initial value for the dependence parameter
        initial_alpha21 = copulaparam(copula, Rho(2,1));
        theta0 = [theta0, log(initial_alpha21)]';
        
        Options = optimset('TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'off');
%         gs = GlobalSearch('NumStageOnePoints', 1e3, 'NumTrialPoints', 1e3, 'Display', 'off');
%         problem = createOptimProblem('fmincon', 'x0', theta0, 'objective',@(theta)Loglikelihood_Archimedean_vine_copula_SFA_model(theta, n_inputs, n_corr_terms, y, X, P, us_Sxn, copula),'options', Options);
%         [theta, Logl] = run(gs, problem);
        [theta, Logl] = fminunc(@(theta)Loglikelihood_Archimedean_vine_copula_SFA_model(theta, n_inputs, n_corr_terms, y, X, P, us_Sxn, copula), theta0, Options);
        
        %tranform scale parameters back to true range
        theta(1) = exp(theta(1));
        theta(2:n_inputs+1) = 1./(1+exp(-theta(2:n_inputs+1))); %inverse logit transform of betas
        theta(n_inputs+2:(n_inputs+1)+2+(n_inputs-1)) = exp(theta(n_inputs+2:(n_inputs+1)+2+(n_inputs-1))); %exponential transform of sigma2_u, sigma2_v and sigma2_W

        copula_dependence_params = theta(length(theta)-n_corr_terms+1:end); %last n_corr_terms elements are the copula dependence parameters
        copula_dependence_params(1:n_inputs-1) = exp(copula_dependence_params(1:n_inputs-1)); %exponential transform of bivariate unconditional Archimedian copula dependence params

        theta(length(theta)-n_corr_terms+1:end) = copula_dependence_params;
    else
        if n_inputs == 3
            %Initial value for the dependence parameters - taken as the Archimedian copula parameter implied by the true correlation parameters
            initial_alpha21 = copulaparam(copula, Rho(2,1));
            initial_alpha31 = copulaparam(copula, Rho(3,1));
            initial_rho32_conditional_x1 = Rho(3, 2);
            initial_rho32_conditional_x1_log_form = direct_mapping_mat([1,initial_rho32_conditional_x1; initial_rho32_conditional_x1,1]);
            theta0 = [theta0, log(initial_alpha21), log(initial_alpha31), initial_rho32_conditional_x1_log_form]';

            Options = optimset('TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'off');
%             gs = GlobalSearch('NumStageOnePoints', 1e3, 'NumTrialPoints', 1e3, 'Display', 'off');
%             problem = createOptimProblem('fmincon', 'x0', theta0, 'objective',@(theta)Loglikelihood_Archimedean_vine_copula_SFA_model(theta, n_inputs, n_corr_terms, y, X, P, us_Sxn, copula),'options', Options);
%             [theta, Logl] = run(gs, problem);
            [theta, Logl] = fminunc(@(theta)Loglikelihood_Archimedean_vine_copula_SFA_model(theta, n_inputs, n_corr_terms, y, X, P, us_Sxn, copula), theta0, Options);

        elseif n_inputs == 4
            %Initial value for the dependence parameters - taken as the Archimedian copula parameter implied by the true correlation parameters
            initial_alpha21 = copulaparam(copula, Rho(2,1));
            initial_alpha31 = copulaparam(copula, Rho(3,1));
            initial_alpha41 = copulaparam(copula, Rho(4,1));
            initial_rho32_conditional_x1 = Rho(3, 2);
            initial_rho42_conditional_x1 = Rho(4, 2);
            initial_rho43_conditional_x1x2 = Rho(4, 3);
            initial_rho32_conditional_x1_log_form = direct_mapping_mat([1,initial_rho32_conditional_x1; initial_rho32_conditional_x1,1]);
            initial_rho42_conditional_x1_log_form = direct_mapping_mat([1,initial_rho42_conditional_x1; initial_rho42_conditional_x1,1]);
            initial_rho42_conditional_x1x2_log_form = direct_mapping_mat([1,initial_rho43_conditional_x1x2; initial_rho43_conditional_x1x2,1]);
            theta0 = [theta0, log(initial_alpha21), log(initial_alpha31),  log(initial_alpha41), initial_rho32_conditional_x1_log_form, initial_rho42_conditional_x1_log_form, initial_rho42_conditional_x1x2_log_form]';

            Options = optimset('TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 20000, 'MaxFunEvals', 6 * 20000, 'Display', 'off');
%             gs = GlobalSearch('NumStageOnePoints', 1e3, 'NumTrialPoints', 1e3, 'Display', 'off');
%             problem = createOptimProblem('fmincon', 'x0', theta0, 'objective',@(theta)Loglikelihood_Archimedean_vine_copula_SFA_model(theta, n_inputs, n_corr_terms, y, X, P, us_Sxn, copula),'options', Options);
%             [theta, Logl] = run(gs, problem);
            [theta, Logl] = fminunc(@(theta)Loglikelihood_Archimedean_vine_copula_SFA_model(theta, n_inputs, n_corr_terms, y, X, P, us_Sxn, copula), theta0, Options);
        end   
        %tranform scale parameters back to true range
        theta(1) = exp(theta(1));
        theta(2:n_inputs+1) = 1./(1+exp(-theta(2:n_inputs+1))); %inverse logit transform of betas
        theta(n_inputs+2:(n_inputs+1)+2+(n_inputs-1)) = exp(theta(n_inputs+2:(n_inputs+1)+2+(n_inputs-1))); %exponential transform of sigma2_u, sigma2_v and sigma2_W

        copula_dependence_params = theta(length(theta)-n_corr_terms+1:end); %last n_corr_terms elements are the copula dependence parameters
        copula_dependence_params(1:n_inputs-1) = exp(copula_dependence_params(1:n_inputs-1)); %exponential transform of bivariate unconditional Archimedian copula dependence params

        %inverse transform the Gaussian copula correlation parameters
        rhos_log_form = copula_dependence_params(n_inputs:end);
        rhos = zeros(length(rhos_log_form), 1);
        for i=1:length(rhos_log_form)
            Rho = inverse_mapping_vec(rhos_log_form(i));
            rhos(i) = Rho(2,1);
        end
        theta(length(theta)-n_corr_terms+1:end) = [copula_dependence_params(1:n_inputs-1); rhos];
    end
end