library('grf')
library('parallel')
library('caret')
library('glmnet')

project_filepath = 'C://Users//Robert James//Dropbox (Sydney Uni)//Estimating Technical Inefficiency Project'
n_cores = detectCores(all.tests = FALSE, logical = TRUE)

### Hyper-parameter optimization ###
  #Import data
train_eps = as.matrix(read.csv(sprintf('%s//panel_SFA_application_Results//panel_SFA_RS2007_electricty_application_NN_train_eps.csv', project_filepath), header = TRUE))
train_u = as.matrix(read.csv(sprintf('%s//panel_SFA_application_Results//panel_SFA_RS2007_electricty_application_NN_train_u.csv', project_filepath), header = TRUE))
N = length(train_u)
T = ncol(train_u)

#Hyper-parameter grid search space
num_trees_search_space = c(2000, 3000, 4000)
min_node_size_search_space = c(50, 100, 200, 300, 500, 1000)
CV_data_matrix = cbind(train_u, train_eps)
CV_loss_matrix = matrix(ncol = 3)
split_idx = createFolds(train_u[1:nrow(train_u), 1], k = 5, list = FALSE, returnTrain = FALSE)

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
        #The ll.lambda (ridge penalty for prediction is tuned automatically)
        LL_forest_object_t = ll_regression_forest(train_fold_X, train_fold_yt,
                                                  num.trees = num_trees,
                                                  num.threads = n_cores,
                                                  min.node.size = min_node_size,
                                                  mtry = T,
                                                  enable.ll.split = TRUE)
        u_hat_t = predict(LL_forest_object_t, test_fold_X)$predictions
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

#Import observed empirical data 
test_eps = as.matrix(read.csv(sprintf('%s//panel_SFA_application_Results//panel_SFA_RS2007_electricty_application_NN_test_eps.csv', project_filepath), header = TRUE))
N = 72
T = ncol(test_eps)

u_hat = matrix(NaN, N, T)
V_u_hat = matrix(NaN, N, T)
#Fit a LL forest model to predict u_it using all eps_i1,  ..., eps_iT
for (t in c(1:T)){
  train_ut = train_u[1:nrow(train_u), t]
  
  #Pilot Lasso to select covariates for the local linear regression
  lasso.mod <- cv.glmnet(train_eps, train_ut, alpha = 1)
  lasso.coef <- predict(lasso.mod, type = "nonzero")
  selected <- lasso.coef[,1]

  #The ll.lambda (ridge penalty for prediction is tuned automatically)
  LL_forest_object_t = ll_regression_forest(train_eps, train_ut,
                                            num.trees = best_params[1],
                                            min.node.size = best_params[2],
                                            num.threads = n_cores,
                                            mtry = T,
                                            enable.ll.split = TRUE,
                                            seed = 10)
  print(variable_importance(LL_forest_object_t))
  #remove rows with nan
  n_NaNs = N - sum(complete.cases(test_eps))
  test_eps = test_eps[complete.cases(test_eps),]
  
  results_t.llf.var = predict(LL_forest_object_t, test_eps, estimate.variance = TRUE, linear.correction.variables = selected)
  u_hat_t = results_t.llf.var$predictions
  V_u_hat_t = results_t.llf.var$variance.estimates
  u_hat[1:length(u_hat_t), t] = u_hat_t
  V_u_hat[1:length(V_u_hat_t), t] = V_u_hat_t
}

#Save the MSE in the simulation results file 
write.csv(u_hat, sprintf('%s//panel_SFA_application_Results//RS2007_electricity_LLF_Gaussian_copula_u_hat.csv', project_filepath), row.names = FALSE)
write.csv(V_u_hat, sprintf('%s//panel_SFA_application_Results//RS2007_electricity_LLF_Gaussian_copula_V_u_hat.csv', project_filepath), row.names = FALSE)
