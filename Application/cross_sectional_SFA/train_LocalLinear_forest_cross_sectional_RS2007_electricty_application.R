#install.packages('grf', repos="http://cran.rstudio.com/")
library('grf')
library('parallel')
library('caret')

project_filepath = 'C://Users//Robert James//Dropbox (Sydney Uni)//Estimating Technical Inefficiency Project'
n_cores = detectCores(all.tests = FALSE, logical = TRUE)

### Hyper-parameter optimization ###
#Import data
train_eps_W = as.matrix(read.csv(sprintf('%s//cross_sectional_SFA_application_Results//cross_sectional_SFA_RS2007_electricity_application_NN_train_eps_W.csv', project_filepath), header = TRUE))
train_u = as.matrix(read.csv(sprintf('%s//cross_sectional_SFA_application_Results//cross_sectional_SFA_RS2007_electricity_application_NN_train_u.csv', project_filepath), header = TRUE))
n = length(train_u)
n_inputs = 3

#5-fold cross validation 
#Hyper-parameter grid search space
num_trees_search_space = c(2000, 3000, 4000)
min_node_size_search_space = c(50, 100, 200, 300, 500, 1000)
CV_data_matrix = cbind(train_u, train_eps_W)
CV_loss_matrix = matrix(ncol = 3)
split_idx = createFolds(train_u, k = 5, list = FALSE, returnTrain = FALSE)

#Loop over the hyper-parameter search space
i = 1
for (num_trees in num_trees_search_space){
  for (min_node_size in min_node_size_search_space){
    CV_losses = vector()
    for (fold in c(1:5)){
      test_idx = which(split_idx == fold)
      train_idx = which(split_idx != fold)
      
      train_fold_y = CV_data_matrix[train_idx,1]
      train_fold_X = CV_data_matrix[train_idx,2:ncol(CV_data_matrix)]
      test_fold_y = CV_data_matrix[test_idx,1]
      test_fold_X = CV_data_matrix[test_idx,2:ncol(CV_data_matrix)]
      
      #Fit a LL forest model 
      LL_forest_object = ll_regression_forest(train_fold_X, train_fold_y,
                                              num.trees = num_trees,
                                              num.threads = n_cores,
                                              min.node.size = min_node_size,
                                              mtry = n_inputs, 
                                              enable.ll.split = TRUE)
      u_hat = predict(LL_forest_object, test_fold_X)$predictions
      fold_loss = mean((u_hat - test_fold_y)^2)
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

#Import data
test_eps_W = as.matrix(read.csv(sprintf('%s//cross_sectional_SFA_application_Results//cross_sectional_SFA_RS2007_electricity_application_NN_test_eps_W.csv', project_filepath), header = TRUE))

#Fit a LL forest model 
LL_forest_object = ll_regression_forest(train_eps_W, train_u,
                                        num.trees = best_params[1],
                                        min.node.size = best_params[2],
                                        num.threads = n_cores,
                                        mtry = n_inputs,
                                        enable.ll.split = TRUE,
                                        seed = 1234
)

results.llf.var = predict(LL_forest_object, test_eps_W, estimate.variance = TRUE)
u_hat = results.llf.var$predictions
V_u_hat = results.llf.var$variance.estimates

results = cbind(u_hat, V_u_hat)

#Save the MSE in the simulation results file 
write.csv(results, sprintf('%s//cross_sectional_SFA_application_Results//LLF_Gaussian_copula_u_hat.csv', project_filepath), row.names = FALSE)
