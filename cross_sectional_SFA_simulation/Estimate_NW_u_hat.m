function [u_hat, V_u_hat] = Estimate_NW_u_hat(theta, n_inputs, n_corr_terms, y, X, U_hat, S_kernel)
    n = length(y);
    
    alpha = theta(1);
    beta = theta(2:n_inputs+1);
    sigma2_v = theta(2+n_inputs); 
    sigma2_u = theta(3+n_inputs);
  
    %Observed variables
        %Cobb-Douglas so alpha is logged
    obs_eps = y - log(alpha) - X*beta; %composed errors from the production function equation (i.e residuals from the production function)
    rep_obs_eps = reshape(repelem(obs_eps, S_kernel), S_kernel, n);

    %Simulated variables
    simulated_v = normrnd(0, sqrt(sigma2_v), S_kernel, 1);
    simulated_u = sqrt(sigma2_u)*norminv((U_hat(:,1)+1)/2, 0,1); %simulated half normal rvs
    simulated_eps = simulated_v - simulated_u; %Construct simulated eps (v-u)

    %Bandwidth information for each conditioning variable
        %rule of thumb bandwidth estimator
    h_rot = 1.06*S_kernel^(-1/5)*(max(std(simulated_eps), iqr(simulated_eps)/1.34));
        %Leave one out CV bandwidth selection
%     h_trial_points = linspace(h_rot/3, h_rot*3, 50)';
%     all_h_CV_results = zeros(length(h_trial_points), 2);
%     all_h_CV_results(:,1) = h_trial_points;
%     parfor j=1:length(h_trial_points)
%         h_trial = h_trial_points(j);
%      %leave-one-out prediction errors
%         leave_one_out_fitted_values = zeros(S_kernel,1);
%         for i = 1:S_kernel
%             leave_one_out_simulated_eps = simulated_eps;
%             leave_one_out_simulated_eps(i) = [];
%             leave_one_out_simulated_u = simulated_u;
%             leave_one_out_simulated_u(i) = [];
% 
%             leave_one_out_eps_kernel = normpdf((leave_one_out_simulated_eps - simulated_eps(i))./h_trial);
%             leave_one_out_kernel_product = leave_one_out_eps_kernel;
%             sum((leave_one_out_kernel_product./sum(leave_one_out_kernel_product)).*leave_one_out_simulated_u);
%             leave_one_out_fitted_values(i,1) = sum(leave_one_out_kernel_product.*leave_one_out_simulated_u)/sum(leave_one_out_kernel_product);
%         end
%         leave_one_out_residuals = (simulated_u - leave_one_out_fitted_values);
%         all_h_CV_results(j,2) = mean(leave_one_out_residuals.^2);
%     end
%     [~, row_idx] = min(all_h_CV_results(:,2));
%     h_cv = all_h_CV_results(row_idx, 1);
    
    %Compute kernel estimates for E[u|eps]
    kernel_regression_results1 = zeros(n,1);
    kernel_regression_results2 = zeros(n,1);
    for i=1:n
        eps_kernel = normpdf((simulated_eps - rep_obs_eps(:,i))./h_rot);
        kernel_product = eps_kernel;
        kernel_regression_results1(i,1) = sum(kernel_product.*simulated_u)/sum(kernel_product);
        kernel_regression_results2(i,1) = sum(kernel_product.*(simulated_u.^2))/sum(kernel_product);
    end
    u_hat = kernel_regression_results1;
    
    %Compute kernel estimates for variance(E[u|eps])
    V_u_hat = kernel_regression_results2 - (u_hat.^2);
   
end