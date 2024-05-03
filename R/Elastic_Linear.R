#' ---------------------------------------------------------------------------------------------------------------
#' Fit Elastic Net Linear Regression Model
#' ---------------------------------------------------------------------------------------------------------------
#' 
#' This function fits an Elastic Net linear regression model to the given dataset and evaluates its accuracy.
#'
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector specifying the predictor variables.
#' @param response A character string specifying the response variable.
#' @param lambda A numeric value specifying the regularization parameter lambda (default: 1).
#' @param top_k An integer specifying the number of top features to select (optional).
#'
#' @return A list containing the fitted model, R-squared on training data, R-squared on test data, and selected predictors.
#'
#' @examples
#' data <- read.csv("data.csv")
#' predictors <- c("predictor1", "predictor2", "predictor3")
#' response <- "response"
#' fit_linear_elastic_net(data, predictors, response)
#'
#' @import glmnet
#' @importFrom base cat
#' @importFrom base sum
#' @importFrom base mean
#' @importFrom base print
#' @importFrom stats na.omit
#' @importFrom stats is.data.frame
#' @importFrom stats saveRDS
#' @importFrom stats as.matrix
#' @export
#' 

fit_linear_elastic_net <- function(data, predictors, response, lambda = 1, top_k = NULL) {

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
  elastic_model <- glmnet(x_train, y_train, alpha = 0.5, lambda = lambda)

  saveRDS(elastic_model, "elatic_net.rds")


  # Predict and calculate R-squared for the training data
  predictions_train <- predict(elastic_model, s = lambda, newx = x_train)
  r_squared_train <- 1 - sum((y_train - predictions_train)^2) / sum((y_train - mean(y_train))^2)

  # Predict and calculate R-squared for the test data
  predictions_test <- predict(elastic_model, s = lambda, newx = x_test)
  r_squared_test <- 1 - sum((y_test - predictions_test)^2) / sum((y_test - mean(y_test))^2)

  #print(elastic_model)
  #print(summary(elastic_model))
  cat("R-squared on train data:", r_squared_train, "\n")
  cat("R-squared on test data:", r_squared_test, "\n")
  print("Ridge model is saved in present directory with the name elatic_net")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")

  return(list(model = elastic_model, r_squared_train = r_squared_train, r_squared_test = r_squared_test, predictors = predictors))
}