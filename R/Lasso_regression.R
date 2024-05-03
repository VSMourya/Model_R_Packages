
#' ---------------------------------------------------------------------------------------------------------------
#' Function to fit Lasso Linear Regression and evaluate accuracy
#' ---------------------------------------------------------------------------------------------------------------
#' 
#' This function fits a linear regression model using Lasso regularization and evaluates its accuracy on both training and testing datasets.
#' 
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector specifying the predictor variables.
#' @param response A character string specifying the response variable.
#' @param lambda A numeric value specifying the regularization parameter lambda. Default is 1.
#' @param top_k An integer specifying the number of top features to select. Default is NULL.
#' 
#' @return A list containing the fitted Lasso linear regression model, R-squared on the training data, R-squared on the test data, and the selected predictors.
#' 
#' @examples
#' data <- read.csv("data.csv")
#' predictors <- c("predictor1", "predictor2", "predictor3")
#' response <- "response"
#' fit_linear_lasso_regression(data, predictors, response)
#' 
#' @importFrom glmnet glmnet
#' @importFrom stats predict
#' @importFrom stats print
#' @importFrom base cat
#' @export
  
  fit_linear_lasso_regression <- function(data, predictors, response, lambda = 1, top_k = NULL) {

  if (!require(glmnet, quietly = TRUE)) {
    install.packages("glmnet")
    library(glmnet)
  } else {
    cat("Package 'glmnet' is already installed.\n")
  }

  data <- na.omit(data)
  if (!is.data.frame(data)) {
    stop("Data must be a dataframe")
  }
  if (!(response %in% names(data))) {
    stop("Predictor or response not found in the dataframe")
  }

  if (!is.null(top_k)) {
    predictors <- select_top_features(data = data, predictors = predictors , response = response, k = top_k)
    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    print("Selected top K Features")
    print(predictors)
  }

  split_result <- train_test_split(data)
  data_train <- split_result$train
  data_test <- split_result$test

  cat("Dimensions of Training Data - Rows:", nrow(data_train), "Columns:", ncol(data_train), "\n")
  cat("Dimensions of Testing Data - Rows:", nrow(data_test), "Columns:", ncol(data_test), "\n")

  print("Starting building model")

  # Prepare data for glmnet
  y_train <- data_train[[response]]
  x_train <- as.matrix(data_train[predictors])

  y_test <- data_test[[response]]
  x_test <- as.matrix(data_test[predictors])

  # Fit Ridge Regression model
  lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = lambda)

  saveRDS(lasso_model, "lasso_regression.rds")

  # Predict and calculate R-squared for the training data
  predictions_train <- predict(lasso_model, s = lambda, newx = x_train)
  r_squared_train <- 1 - sum((y_train - predictions_train)^2) / sum((y_train - mean(y_train))^2)

  # Predict and calculate R-squared for the test data
  predictions_test <- predict(lasso_model, s = lambda, newx = x_test)
  r_squared_test <- 1 - sum((y_test - predictions_test)^2) / sum((y_test - mean(y_test))^2)

  #print(lasso_model)
  #print(summary(lasso_model))
  cat("R-squared on train data:", r_squared_train, "\n")
  cat("R-squared on test data:", r_squared_test, "\n")
  print("Lasso model is saved in present directory with the name lasso_regression")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")

  return(list(model = lasso_model, r_squared_train = r_squared_train, r_squared_test = r_squared_test, predictors = predictors))
}
