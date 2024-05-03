#' ---------------------------------------------------------------------------------------------------------------
#' Fit Boosted Trees Logistic Regression Model
#' ---------------------------------------------------------------------------------------------------------------
#' 
#' This function fits a Boosted Trees logistic regression model to the given dataset and evaluates its accuracy.
#'
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector specifying the predictor variables.
#' @param response A character string specifying the binary response variable.
#' @param top_k An integer specifying the number of top features to select (optional).
#'
#' @return A list containing the fitted model, accuracy on training data, accuracy on test data, and selected predictors.
#'
#' @examples
#' data <- read.csv("data.csv")
#' predictors <- c("predictor1", "predictor2", "predictor3")
#' response <- "response"
#' fit_boosted_trees_binary(data, predictors, response)
#'
#' @import xgboost
#' @importFrom stats mean
#' @importFrom base cat
#'
#' @export

fit_boosted_trees_binary <- function(data, predictors, response, top_k = NULL) {
  if (!require(xgboost, quietly = TRUE)) {
    install.packages("xgboost")
    library(xgboost)
  } else {
    cat("Package 'xgboost' is already installed.\n")
  }

  #data[[response]] <- as.numeric(factor(data[[response]], levels = c(0, 1)))

  # Check if response variable is binary
  if (!is.factor(data[[response]]) || length(levels(data[[response]])) != 2) {
    stop("Response variable must be binary (factor with 2 levels)")
  }

  if (!is.null(top_k)) {
    predictors <- select_top_features(data = data, predictors = predictors, response = response, k = top_k)
    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    print("Selected top K Features")
    print(predictors)
  }

  # Split data into training and testing sets
  split_result <- train_test_split(data)
  data_train <- split_result$train
  data_test <- split_result$test

  cat("Dimensions of Training Data - Rows:", nrow(data_train), "Columns:", ncol(data_train), "\n")
  cat("Dimensions of Testing Data - Rows:", nrow(data_test), "Columns:", ncol(data_test), "\n")

  # Convert data to DMatrix format
  dtrain <- xgb.DMatrix(data = as.matrix(data_train[predictors]), label = data_train[[response]])
  dtest <- xgb.DMatrix(data = as.matrix(data_test[predictors]), label = data_test[[response]])

  # Set parameters for XGBoost model
  params <- list(
    objective = "binary:logistic",
    eval_metric = "error",
    eta = 0.01,
    max_depth = 4,
    subsample = 0.5
  )

  # Train the XGBoost model
  boosted_model <- xgboost(params = params, data = dtrain, nrounds = 100, verbose = 0)

  # Save the model
  xgb.save(boosted_model, "boosted_trees_binary.model")

  # Predict on training data
  train_predictions <- predict(boosted_model, dtrain)
  train_accuracy <- mean((train_predictions > 0.5) == data_train[[response]])
  cat("Accuracy on training data:", train_accuracy, "\n")

  # Predict on test data
  test_predictions <- predict(boosted_model, dtest)
  test_accuracy <- mean((test_predictions > 0.5) == data_test[[response]])
  cat("Accuracy on test data:", test_accuracy, "\n")
  print("Boosted Trees model is saved in the present directory with the name boosted_trees_binary.model")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")

  # Return the model and accuracies
  return(list(model = boosted_model, accuracy_train = train_accuracy, accuracy_test = test_accuracy, predictors = predictors))
}
