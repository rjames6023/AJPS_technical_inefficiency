library('grf')
library('parallel')
library('caret')

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
n = as.numeric(args[1])
rho_u_w1 = args[2]
rho_u_w2 = args[3]
rho_u_w3 = args[4]
rho_w1_w2 = args[5]
rho_w1_w3 = args[6]
rho_w2_w3 = args[7]
copula = args[8]
export_string = args[9]

if ((rho_u_w1 != '~') && (rho_u_w2 == '~') && (rho_u_w3  == '~')) {
  n_inputs = 2
  rho_u_W = as.numeric(c(rho_u_w1))
  rho_W = c()
} else if ((rho_u_w1 != '~') && (rho_u_w2 != '~') && (rho_u_w3  == '~')) {
  n_inputs = 3;
  rho_u_W = as.numeric(c(rho_u_w1, rho_u_w2))
  rho_W = as.numeric(c(rho_w1_w2))
} else if ((rho_u_w1 != '~') && (rho_u_w2 != '~') && (rho_u_w3 != '~')) {
  n_inputs = 4;
  rho_u_W = as.numeric(c(rho_u_w1, rho_u_w2, rho_u_w3))
  rho_W = as.numeric(c(rho_w1_w2, rho_w1_w3, rho_w2_w3))
}

project_filepath = '//project//RDS-FOB-Eff_Dens_1-RW//technical_inefficiency_estimation'
n_cores = detectCores(all.tests = FALSE, logical = TRUE)

#Unzip data files
rho_folder_name = paste(format(round(rho_u_W, 3), nsmall = 3), collapse = " ")
uzp = sprintf('%s//Datasets//simulation_data//n_inputs=%d//n=%d//rho=%s.zip', project_filepath, n_inputs, n, rho_folder_name)
dir.create(sprintf('%s//Datasets//simulation_data//n_inputs=%d//n=%d//rho=%s', project_filepath, n_inputs, n, rho_folder_name))
unzip(uzp, exdir = sprintf('%s//Datasets//simulation_data//n_inputs=%d//n=%d//rho=%s', project_filepath, n_inputs, n, rho_folder_name))

#Pre-allocate CV training data matrices
CV_train_eps_W = matrix(0, 10000, n_inputs)
CV_train_u = matrix(0, 10000, 1)
for (c in 1:500){
  #Import data
  train_eps_W = as.matrix(read.csv(sprintf('%s//Datasets//simulation_data//n_inputs=%d//n=%d//rho=%s//rho=%s//%s_simulation=%d_NN_train_eps_W.csv', project_filepath, n_inputs, n, rho_folder_name, rho_folder_name, export_string, c), header = TRUE))
  train_u = as.matrix(read.csv(sprintf('%s//Datasets//simulation_data//n_inputs=%d//n=%d//rho=%s//rho=%s//%s_simulation=%d_NN_train_u.csv', project_filepath, n_inputs, n, rho_folder_name, rho_folder_name, export_string, c), header = TRUE))

  random_idx = sample(c(1:10000), 10000/500)
  random_train_eps_W = train_eps_W[random_idx,]
  random_train_u = as.matrix(train_u[random_idx,])
  
  if (c == 1){
    colnames(CV_train_eps_W) = colnames(train_eps_W)
    colnames(CV_train_u) = colnames(train_u)
    CV_train_eps_W[1:20,1:n_inputs] = random_train_eps_W
    CV_train_u[1:20,1] = random_train_u
  } else{
    CV_train_eps_W[(((c-1)*20)+1):(c*20), 1:n_inputs] = random_train_eps_W
    CV_train_u[(((c-1)*20)+1):(c*20), 1] = random_train_u
  }
}

#5-fold cross validation 
  #Hyper-parameter grid search space
num_trees_search_space = c(2000, 3000, 4000)
min_node_size_search_space = c(50, 100, 300, 500, 1000)
CV_data_matrix = cbind(CV_train_u, CV_train_eps_W)
CV_loss_matrix = matrix(ncol = 3)
split_idx = createFolds(CV_train_u, k = 5, list = FALSE, returnTrain = FALSE)

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
      u_hat = predict(LL_forest_object, test_fold_X, ll.lambda = 0.1)$predictions
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

LLF_murphy_scores_matrix = matrix(nrow = 501, ncol = 500)
u_hat_MSE_results = vector()
u_hat_upper_quartile_MSE_results = vector()
u_hat_lower_quartile_MSE_results = vector()
u_hat_mid_quartile_MSE_results = vector()

for (c in 1:500){
  #Import data
  train_eps_W = as.matrix(read.csv(sprintf('%s//Datasets//simulation_data//n_inputs=%d//n=%d//rho=%s//rho=%s//%s_simulation=%d_NN_train_eps_W.csv', project_filepath, n_inputs, n, rho_folder_name, rho_folder_name, export_string, c), header = TRUE))
  train_u = as.matrix(read.csv(sprintf('%s//Datasets//simulation_data//n_inputs=%d//n=%d//rho=%s//rho=%s//%s_simulation=%d_NN_train_u.csv', project_filepath, n_inputs, n, rho_folder_name, rho_folder_name, export_string, c), header = TRUE))
  test_eps_W = as.matrix(read.csv(sprintf('%s//Datasets//simulation_data//n_inputs=%d//n=%d//rho=%s//rho=%s//%s_simulation=%d_NN_test_eps_W.csv', project_filepath, n_inputs, n, rho_folder_name, rho_folder_name, export_string, c), header = TRUE))
  test_u = as.matrix(read.csv(sprintf('%s//Datasets//simulation_data//n_inputs=%d//n=%d//rho=%s//rho=%s//%s_simulation=%d_NN_test_u.csv', project_filepath, n_inputs, n, rho_folder_name, rho_folder_name, export_string, c), header = TRUE))
  
  #Fit a LL forest model 
  LL_forest_object = ll_regression_forest(train_eps_W, train_u,
                                          num.trees = best_params[1],
                                          min.node.size = best_params[2],
                                          num.threads = n_cores,
                                          mtry = n_inputs,
                                          enable.ll.split = TRUE,
                                          seed = c
  )
  
  u_hat = predict(LL_forest_object, test_eps_W)$predictions
  MSE = mean((u_hat - test_u)^2)
  u_hat_MSE_results[c] = MSE
  
  #MSE of upper/lower quartile TI scores
  upper_idx = which(test_u >= quantile(test_u, 0.75))
  upper_quartile_MSE = mean((u_hat[upper_idx] - test_u[upper_idx])^2)
  u_hat_upper_quartile_MSE_results[c] = upper_quartile_MSE
  
  mid_idx = which(test_u < quantile(test_u, 0.75) & test_u > quantile(test_u, 0.25))
  mid_quartile_MSE =mean((u_hat[mid_idx] - test_u[mid_idx])^2)
  u_hat_mid_quartile_MSE_results[c] = upper_quartile_MSE
  
  lower_idx = which(test_u<= quantile(test_u, 0.25))
  lower_quartile_MSE = mean((u_hat[lower_idx] - test_u[lower_idx])^2)
  u_hat_lower_quartile_MSE_results[c] = lower_quartile_MSE
  
  murphydiagram_results = murphydiagram_scores(test_u, u_hat)
  LLF_murphy_scores_matrix[,c] = murphydiagram_results$s1ave
}

MSE = mean(u_hat_MSE_results)
upper_quartile_MSE = mean(upper_quartile_MSE)
lower_quartile_MSE = mean(lower_quartile_MSE)
mid_quartile_MSE = mean(mid_quartile_MSE)

#Murphy Diagram scores
LLF_final_murphy_scores = data.frame(cbind(murphydiagram_results$theta, rowMeans(LLF_murphy_scores_matrix)))
colnames(LLF_final_murphy_scores) = c('Theta', 'LLF_elementary_score')

#Delete the unzipped raw data file
unlink(sprintf('%s//Datasets//simulation_data//n_inputs=%d//n=%d//rho=%s', project_filepath, n_inputs, n, rho_folder_name), recursive = TRUE)

#Save the MSE in the simulation results file 
if (copula == '~'){
  results_file = read.csv(sprintf('%s//Results//n_inputs=%d//u_hat_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, n_inputs, rho_folder_name, n_inputs, n), header = TRUE)
  results_file$local_linear_regression_tree = MSE
  write.csv(results_file, sprintf('%s//Results//n_inputs=%d//u_hat_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, n_inputs, rho_folder_name, n_inputs, n), row.names = FALSE)

  upper_quartile_results_file = read.csv(sprintf('%s//Results//n_inputs=%d//u_hat_upper_quartile_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, n_inputs, rho_folder_name, n_inputs, n), header = TRUE)
  upper_quartile_results_file$local_linear_regression_tree = upper_quartile_MSE
  write.csv(upper_quartile_results_file, sprintf('%s//Results//n_inputs=%d//u_hat_upper_quartile_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, n_inputs, rho_folder_name, n_inputs, n), row.names = FALSE)
  
  lower_quartile_results_file = read.csv(sprintf('%s//Results//n_inputs=%d//u_hat_lower_quartile_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, n_inputs, rho_folder_name, n_inputs, n), header = TRUE)
  lower_quartile_results_file$local_linear_regression_tree = lower_quartile_MSE
  write.csv(lower_quartile_results_file, sprintf('%s//Results//n_inputs=%d//u_hat_lower_quartile_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, n_inputs, rho_folder_name, n_inputs, n), row.names = FALSE)

  mid_quartile_results_file = read.csv(sprintf('%s//Results//n_inputs=%d//u_hat_mid_quartile_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, n_inputs, rho_folder_name, n_inputs, n), header = TRUE)
  mid_quartile_results_file$local_linear_regression_tree = mid_quartile_MSE
  write.csv(mid_quartile_results_file, sprintf('%s//Results//n_inputs=%d//u_hat_mid_quartile_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, n_inputs, rho_folder_name, n_inputs, n), row.names = FALSE)

  write.csv(LLF_final_murphy_scores, sprintf('%s//Results//n_inputs=%d//LLF_Murphy_Diagram_LLF_results rho=%s n_inputs=%d n=%d.csv', project_filepath, n_inputs, rho_folder_name, n_inputs, n), row.names = FALSE)
}else{
  results_file = read.csv(sprintf('%s//%s//Results//n_inputs=%d//u_hat_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, copula, n_inputs, rho_folder_name, n_inputs, n), header = TRUE)
  results_file$local_linear_regression_tree = MSE
  write.csv(results_file, sprintf('%s//%s//Results//n_inputs=%d//u_hat_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, copula, n_inputs, rho_folder_name, n_inputs, n), row.names = FALSE)
  
  upper_quartile_results_file = read.csv(sprintf('%s//%s//Results//n_inputs=%d//u_hat_upper_quartile_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, copula, n_inputs, rho_folder_name, n_inputs, n), header = TRUE)
  upper_quartile_results_file$local_linear_regression_tree = upper_quartile_MSE
  write.csv(upper_quartile_results_file, sprintf('%s//%s//Results//n_inputs=%d//u_hat_upper_quartile_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, copula, n_inputs, rho_folder_name, n_inputs, n), row.names = FALSE)
  
  lower_quartile_results_file = read.csv(sprintf('%s//%s//Results//n_inputs=%d//u_hat_lower_quartile_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, copula, n_inputs, rho_folder_name, n_inputs, n), header = TRUE)
  lower_quartile_results_file$local_linear_regression_tree = lower_quartile_MSE
  write.csv(lower_quartile_results_file, sprintf('%s//%s//Results//n_inputs=%d//u_hat_lower_quartile_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, copula, n_inputs, rho_folder_name, n_inputs, n), row.names = FALSE)
  
  mid_quartile_results_file = read.csv(sprintf('%s//%s//Results//n_inputs=%d//u_hat_mid_quartile_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, copula, n_inputs, rho_folder_name, n_inputs, n), header = TRUE)
  mid_quartile_results_file$local_linear_regression_tree = mid_quartile_MSE
  write.csv(mid_quartile_results_file, sprintf('%s//%s//Results//n_inputs=%d//u_hat_mid_quartile_MSE_results rho=%s n_inputs=%d n=%d.csv', project_filepath, copula, n_inputs, rho_folder_name, n_inputs, n), row.names = FALSE)
  
  write.csv(LLF_final_murphy_scores, sprintf('%s//%s//Results//n_inputs=%d//LLF_Murphy_Diagram_LLF_results rho=%s n_inputs=%d n=%d.csv', project_filepath, copula, n_inputs, rho_folder_name, n_inputs, n), row.names = FALSE)
}