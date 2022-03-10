library('grf')
library('parallel')
library('caret')
library('glmnet')

murphydiagram_scores <- function(y, f1){
  n <- length(y)
  
  max_tmp = sqrt(1)*qnorm((0.999+1)/2, mean = 0, sd = 1) #set the max theta to the 99.9th quantile of the distribution of technical inefficiency
  min_tmp = 0 #Theoretical minimum of technical inefficiency
  tmp <- c(min_tmp-0.1*(max_tmp - min_tmp), max_tmp + 0.1*(max_tmp - min_tmp))
  theta <- seq(tmp[1], tmp[2], l=501)
  
  s1 <- s2 <- matrix(0,nrow=501,ncol=n)*NA
  
  max1 <- pmax(f1,y)
  min1 <- pmin(f1,y)
  
  for(j in 1:n){
    s1[,j] <- abs(y[j]-theta) * (max1[j] > theta) * (min1[j] <= theta)
  }
  s1ave <- rowMeans(s1, na.rm=TRUE)
  return (list(theta = theta, s1ave = s1ave))
}

args = commandArgs(trailingOnly = TRUE)
N = as.numeric(args[1])
T = as.numeric(args[2])
rho1 = as.numeric(args[3])
copula = args[4]
export_string = args[5]

project_filepath = '//project//RDS-FOB-Eff_Dens_1-RW//technical_inefficiency_estimation'
n_cores = detectCores(all.tests = FALSE, logical = TRUE)

#Unzip data files
rho_folder_name = paste(format(round(rho1, 2), nsmall = 2), collapse = " ")
uzp = sprintf('%s//Datasets//panel_simulation_data//N=%d_T=%d//rho=%s.zip', project_filepath, N, T, rho_folder_name)
dir.create(sprintf('%s//Datasets//panel_simulation_data//N=%d_T=%d//rho=%s', project_filepath, N, T, rho_folder_name))
unzip(uzp, exdir = sprintf('%s//Datasets//panel_simulation_data//N=%d_T=%d//rho=%s', project_filepath, N, T, rho_folder_name))

#Pre-allocate CV training data matrices
CV_train_eps = matrix(0, 10000, T)
CV_train_u = matrix(0, 10000, T)
for (c in 1:500){
  #Import data
  train_eps = as.matrix(read.csv(sprintf('%s//Datasets//panel_simulation_data//N=%d_T=%d//rho=%s//rho=%s//panel_simulation=%d_NN_train_eps.csv', project_filepath, N, T, rho_folder_name, rho_folder_name, c), header = TRUE))
  train_u = as.matrix(read.csv(sprintf('%s//Datasets//panel_simulation_data//N=%d_T=%d//rho=%s//rho=%s//panel_simulation=%d_NN_train_u.csv', project_filepath, N, T, rho_folder_name, rho_folder_name, c), header = TRUE))
  
  random_idx = sample(c(1:10000), 10000/500)
  random_train_eps = train_eps[random_idx,]
  random_train_u = train_u[random_idx,]
  
  if (c == 1){
    colnames(CV_train_eps) = colnames(train_eps)
    colnames(CV_train_u) = colnames(train_u)
    CV_train_eps[1:20,1:T] = random_train_eps
    CV_train_u[1:20,1:T] = random_train_u
  } else{
    CV_train_eps[(((c-1)*20)+1):(c*20), 1:T] = random_train_eps
    CV_train_u[(((c-1)*20)+1):(c*20), 1:T] = random_train_u
  }
}

#5-fold cross validation 
#Hyper-parameter grid search space
num_trees_search_space = c(2000, 3000, 4000)
min_node_size_search_space = c(50, 100, 300, 500, 1000)
CV_data_matrix = cbind(CV_train_u, CV_train_eps)
CV_loss_matrix = matrix(ncol = 3)
split_idx = createFolds(CV_train_u[1:nrow(CV_train_u), 1], k = 5, list = FALSE, returnTrain = FALSE)


#Loop over the hyper-parameter search space
i = 1
for (num_trees in num_trees_search_space){
  for (min_node_size in min_node_size_search_space){
    CV_losses = vector()
    for (fold in c(1:5)){
      test_idx = which(split_idx == fold)
      train_idx = which(split_idx != fold)
      
      train_fold_y = CV_data_matrix[train_idx,1:T]
      train_fold_X = CV_data_matrix[train_idx,(T+1):(2*T)]
      test_fold_y = CV_data_matrix[test_idx,1:T]
      test_fold_X = CV_data_matrix[test_idx,(T+1):(2*T)]
      
      u_hat = matrix(0, nrow(test_fold_y), ncol(test_fold_y))
      #Fit a LL forest model to predict u_it using all eps_i1,  ..., eps_iT
      for (t in c(1:T)){
        train_fold_yt = train_fold_y[1:nrow(train_fold_y), t]
        LL_forest_object_t = ll_regression_forest(train_fold_X, train_fold_yt,
                                                num.trees = num_trees,
                                                num.threads = n_cores,
                                                min.node.size = min_node_size,
                                                mtry = T,
                                                enable.ll.split = TRUE)
        u_hat_t = predict(LL_forest_object_t, test_fold_X, ll.lambda = 0.1)$predictions
        u_hat[1:nrow(test_fold_y), t] = u_hat_t
      }
      
      fold_loss = sum((u_hat - test_fold_y)^2)/(N*T)
      CV_losses[fold] = fold_loss
    }
    mean_CV_loss = mean(CV_losses)
    if (i==1){
      CV_loss_matrix[i,1:2] = c(num_trees, min_node_size)
      CV_loss_matrix[i,ncol(CV_loss_matrix)] = mean_CV_loss
    } else{
      CV_loss_matrix = rbind(CV_loss_matrix, c(num_trees, min_node_size, mean_CV_loss))
    }
    i = i+1
  }
}
best_CV_loss = min(CV_loss_matrix[,ncol(CV_loss_matrix)])
best_CV_loss_idx = which(CV_loss_matrix[,ncol(CV_loss_matrix)] == best_CV_loss)
best_params = CV_loss_matrix[best_CV_loss_idx,1:ncol(CV_loss_matrix)-1]
if (!is.null(nrow(best_params))){
  sums = rowSums(best_params)
  best_idx = which(sums == min(sums))
  best_params = best_params[best_idx,]
}

LLF_murphy_scores_matrix = matrix(nrow = 501, ncol = 500)
u_hat_MSE_results = vector()
u_hat_upper_quartile_MSE_results = matrix(nrow = 500, ncol = T)
u_hat_lower_quartile_MSE_results = matrix(nrow = 500, ncol = T)
u_hat_mid_quartile_MSE_results = matrix(nrow = 500, ncol = T)

for (c in 1:500){
  #Import data
  train_eps = as.matrix(read.csv(sprintf('%s//Datasets//panel_simulation_data//N=%d_T=%d//rho=%s//rho=%s//panel_simulation=%d_NN_train_eps.csv', project_filepath, N, T, rho_folder_name, rho_folder_name, c), header = TRUE))
  train_u = as.matrix(read.csv(sprintf('%s//Datasets//panel_simulation_data//N=%d_T=%d//rho=%s//rho=%s//panel_simulation=%d_NN_train_u.csv', project_filepath, N, T, rho_folder_name, rho_folder_name, c), header = TRUE))
  test_eps = as.matrix(read.csv(sprintf('%s//Datasets//panel_simulation_data//N=%d_T=%d//rho=%s//rho=%s//panel_simulation=%d_NN_test_eps.csv', project_filepath, N, T, rho_folder_name, rho_folder_name, c), header = TRUE))
  test_u = as.matrix(read.csv(sprintf('%s//Datasets//panel_simulation_data//N=%d_T=%d//rho=%s//rho=%s//panel_simulation=%d_NN_test_u.csv', project_filepath, N, T, rho_folder_name, rho_folder_name, c), header = TRUE))
  
  u_hat = matrix(0, nrow(test_u), ncol(test_u))
  
  #Fit a LL forest model to predict u_it using all eps_i1,  ..., eps_iT
  for (t in c(1:T)){
    
    train_ut = train_u[1:nrow(train_u), t]
    
    #Pilot Lasso to select covariates for the local linear regression
    lasso.mod <- cv.glmnet(train_eps, train_ut, alpha = 1)
    lasso.coef <- predict(lasso.mod, type = "nonzero")
    selected <- lasso.coef[,1]
    
    LL_forest_object_t = ll_regression_forest(train_eps, train_ut,
                                              num.trees = best_params[1],
                                              min.node.size = best_params[2],
                                              num.threads = n_cores,
                                              mtry = T,
                                              enable.ll.split = TRUE,
                                              seed = c)
    u_hat_t = predict(LL_forest_object_t, test_eps, linear.correction.variables = selected)$predictions
    u_hat[1:nrow(u_hat), t] = u_hat_t
    
    #MSE of upper/lower quartile TI scores
    upper_idx = which(test_u[1:nrow(test_u),t] >= quantile(test_u[1:nrow(test_u),t], 0.75))
    upper_quartile_MSE = mean((u_hat[upper_idx, t] - test_u[upper_idx, t])^2)
    u_hat_upper_quartile_MSE_results[c, t] = upper_quartile_MSE
    
    mid_idx = which(test_u[1:nrow(test_u),t] < quantile(test_u[1:nrow(test_u),t], 0.75) & test_u[1:nrow(test_u),t] > quantile(test_u[1:nrow(test_u),t], 0.25))
    mid_quartile_MSE =mean((u_hat[mid_idx, t] - test_u[mid_idx, t])^2)
    u_hat_mid_quartile_MSE_results[c, t] = upper_quartile_MSE
    
    lower_idx = which(test_u[1:nrow(test_u)]<= quantile(test_u[1:nrow(test_u)], 0.25))
    lower_quartile_MSE = mean((u_hat[lower_idx, t] - test_u[lower_idx, t])^2)
    u_hat_lower_quartile_MSE_results[c, t] = lower_quartile_MSE
    
  }
  
  MSE = sum((u_hat - test_u)^2)/(N*T)
  u_hat_MSE_results[c] = MSE

  murphydiagram_results = murphydiagram_scores(as.vector(test_u), as.vector(u_hat))
  LLF_murphy_scores_matrix[,c] = murphydiagram_results$s1ave
}
SFA_MSE = mean(u_hat_MSE_results)
upper_quartile_MSE = mean(upper_quartile_MSE)
lower_quartile_MSE = mean(lower_quartile_MSE)
mid_quartile_MSE = mean(mid_quartile_MSE)

#Murphy Diagram scores
LLF_final_murphy_scores = data.frame(cbind(murphydiagram_results$theta, rowMeans(LLF_murphy_scores_matrix)))
colnames(LLF_final_murphy_scores) = c('Theta', 'LLF_elementary_score')

#Delete the unzipped raw data file
unlink(sprintf('%s//Datasets//panel_simulation_data//N=%d_T=%d//rho=%s', project_filepath, N, T, rho_folder_name), recursive = TRUE)

#Save the MSE in the simulation results file 
SFA_results_file = read.csv(sprintf('%s//panel_%s_simulation_Results//%s//AR_1=%.2f//u_hat_MSE_results AR_1=%.2f N=%d T=%d.csv', project_filepath, export_string, copula, rho1, rho1, N, T), header = TRUE)
SFA_results_file$local_linear_regression_tree = SFA_MSE
write.csv(SFA_results_file, sprintf('%s//panel_%s_simulation_Results//%s//AR_1=%.2f//u_hat_MSE_results AR_1=%.2f N=%d T=%d.csv', project_filepath, export_string, copula, rho1, rho1, N, T), row.names = FALSE)

upper_quartile_MSE_results_file = read.csv(sprintf('%s//panel_%s_simulation_Results//%s//AR_1=%.2f//u_hat_upper_quartile_MSE_results AR_1=%.2f N=%d T=%d.csv', project_filepath, export_string, copula, rho1, rho1, N, T), header = TRUE)
upper_quartile_MSE_results_file$local_linear_regression_tree = upper_quartile_MSE
write.csv(upper_quartile_MSE_results_file, sprintf('%s//panel_%s_simulation_Results//%s//AR_1=%.2f//u_hat_upper_quartile_MSE_results AR_1=%.2f N=%d T=%d.csv', project_filepath, export_string, copula, rho1, rho1, N, T), row.names = FALSE)

mid_quartile_MSE_results_file = read.csv(sprintf('%s//panel_%s_simulation_Results//%s//AR_1=%.2f//u_hat_mid_quartile_MSE_results AR_1=%.2f N=%d T=%d.csv', project_filepath, export_string, copula, rho1, rho1, N, T), header = TRUE)
mid_quartile_MSE_results_file$local_linear_regression_tree = mid_quartile_MSE
write.csv(mid_quartile_MSE_results_file, sprintf('%s//panel_%s_simulation_Results//%s//AR_1=%.2f//u_hat_mid_quartile_MSE_results AR_1=%.2f N=%d T=%d.csv', project_filepath, export_string, copula, rho1, rho1, N, T), row.names = FALSE)

lower_quartile_MSE_results_file = read.csv(sprintf('%s//panel_%s_simulation_Results//%s//AR_1=%.2f//u_hat_lower_quartile_MSE_results AR_1=%.2f N=%d T=%d.csv', project_filepath, export_string, copula, rho1, rho1, N, T), header = TRUE)
lower_quartile_MSE_results_file$local_linear_regression_tree = lower_quartile_MSE
write.csv(lower_quartile_MSE_results_file, sprintf('%s//panel_%s_simulation_Results//%s//AR_1=%.2f//u_hat_lower_quartile_MSE_results AR_1=%.2f N=%d T=%d.csv', project_filepath, export_string, copula, rho1, rho1, N, T), row.names = FALSE)

write.csv(LLF_final_murphy_scores, sprintf('%s//panel_%s_simulation_Results//%s//AR_1=%.2f//LLF_Murphy_Diagram_LLF_results_%s_simulation AR_1=%.2f N=%d T=%d.csv', project_filepath, export_string, copula, rho1, export_string, rho1, N, T), row.names = FALSE)