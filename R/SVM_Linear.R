#' ----------------------------------------------------------------------------------------------------------------------------------
#' Fit Linear Support Vector Machine (SVM) Model and Evaluate Accuracy
#' ----------------------------------------------------------------------------------------------------------------------------------
#' 
#' This function fits a Linear Support Vector Machine (SVM) model and evaluates its accuracy on both training and testing datasets.
#' 
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector specifying the predictor variables.
#' @param response A character string specifying the response variable.
#' @param top_k An integer specifying the number of top features to select. Default is NULL.
#' 
#' @return A list containing the fitted Linear SVM model, RMSE value on the training data, 
#'         RMSE value on the test data, and the selected predictors.
#' 
#' @examples
#' data <- read.csv("data.csv")
#' predictors <- c("predictor1", "predictor2", "predictor3")
#' response <- "response"
#' fit_linear_svm(data, predictors, response)
#' 
#' @importFrom e1071 svm
#' @importFrom stats predict
#' @importFrom stats mean
#' @importFrom stats sqrt
#' @importFrom base cat
#' @export
fit_linear_svm <- function(data, predictors, response, top_k = NULL) {

  if (!require(e1071, quietly = TRUE)) {
    install.packages("e1071")
    library(e1071)
  } else {
    cat("Package 'e1071' is already installed.\n")
  }

  # Check if response variable is numeric
  if (!is.numeric(data[[response]])) {
    stop("Response variable must be numeric")
  }

  if (!is.null(top_k)) {
    predictors <- select_top_features(data = data, predictors = predictors , response = response, k = top_k)
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

  # Fit Linear SVM model
  svm_model <- svm(as.formula(paste(response, "~", paste(predictors, collapse = "+"))),
                   data = data_train,
                   kernel = "linear")

  # Save the model
  saveRDS(svm_model, "linear_svm.rds")


  # Predict on training data
  train_predictions <- predict(svm_model, data_train)
  train_rmse <- sqrt(mean((train_predictions - data_train[[response]])^2))
  cat("RMSE on training data:", train_rmse, "\n")

  # Predict on test data
  test_predictions <- predict(svm_model, data_test)
  test_rmse <- sqrt(mean((test_predictions - data_test[[response]])^2))
  cat("RMSE on test data:", test_rmse, "\n")
  print("Linear SVM model is saved in the present directory with the name linear_svm.rds")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")

  # Return the model and RMSE values
  return(list(model = svm_model, rmse_train = train_rmse, rmse_test = test_rmse, predictors = predictors))
}
