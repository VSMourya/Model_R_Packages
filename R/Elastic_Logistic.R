#' ---------------------------------------------------------------------------------------------------------------
#' Fit Elastic Net Logistic Regression Model
#' ---------------------------------------------------------------------------------------------------------------
#' 
#' This function fits an Elastic Net logistic regression model to the given dataset and evaluates its accuracy.
#'
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector specifying the predictor variables.
#' @param response A character string specifying the response variable.
#' @param lambda A numeric value specifying the regularization parameter lambda (default: 1).
#' @param top_k An integer specifying the number of top features to select (optional).
#'
#' @return A list containing the fitted model, accuracy on training data, accuracy on test data, and selected predictors.
#'
#' @examples
#' data <- read.csv("data.csv")
#' predictors <- c("predictor1", "predictor2", "predictor3")
#' response <- "response"
#' fit_logistic_elastic_net(data, predictors, response)
#'
#' @import glmnet
#' @importFrom base cat
#' @importFrom base mean
#' @importFrom base print
#' @importFrom stats na.omit
#' @importFrom stats is.data.frame
#' @importFrom stats saveRDS
#' @importFrom stats as.matrix
#' @export

fit_logistic_elastic_net <- function(data, predictors, response, lambda = 1, top_k = NULL) {

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

  # Fit Elastic Net Logistic Regression model
  elastic_model <- glmnet(x_train, y_train, alpha = 0.5, family = "binomial", lambda = lambda)

  saveRDS(elastic_model, "elastic_net_logistic")

  # Predict and calculate accuracy for the training data
  predictions_train <- predict(elastic_model, s = lambda, newx = x_train, type = "response")
  correct_predictions_train <- mean((predictions_train > 0.5) == y_train)

  # Predict and calculate accuracy for the test data
  predictions_test <- predict(elastic_model, s = lambda, newx = x_test, type = "response")
  correct_predictions_test <- mean((predictions_test > 0.5) == y_test)

  cat("Accuracy on train data:", correct_predictions_train, "\n")
  cat("Accuracy on test data:", correct_predictions_test, "\n")
  print("Elastic Net Logistic Regression model is saved in the present directory with the name elastic_net_logistic")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")


  return(list(model = elastic_model, accuracy_train = correct_predictions_train, accuracy_test = correct_predictions_test, predictors = predictors))
}
