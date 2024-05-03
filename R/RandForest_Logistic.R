#' ----------------------------------------------------------------------------------------------------------------------------------
#' Fit Random Forest Logistic Regression Model and Evaluate Accuracy
#' ----------------------------------------------------------------------------------------------------------------------------------
#' 
#' This function fits a Random Forest model with logistic regression and evaluates its accuracy on both training and testing datasets.
#' 
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector specifying the predictor variables.
#' @param response A character string specifying the response variable.
#' @param top_k An integer specifying the number of top features to select. Default is NULL.
#' 
#' @return A list containing the fitted Random Forest model, accuracy on the training data, accuracy on the test data, and the selected predictors.
#' 
#' @examples
#' data <- read.csv("data.csv")
#' predictors <- c("predictor1", "predictor2", "predictor3")
#' response <- "response"
#' fit_random_forest_binary(data, predictors, response)
#' 
#' @importFrom randomForest randomForest
#' @importFrom stats predict
#' @importFrom stats mean
#' @importFrom base cat
#' @export
#' 
fit_random_forest_binary <- function(data, predictors, response, top_k = NULL) {

  if (!require(randomForest, quietly = TRUE)) {
    install.packages("randomForest")
    library(randomForest)
  } else {
    cat("Package 'randomForest' is already installed.\n")
  }

  data[[response]] <- factor(data[[response]], levels = c(0, 1))

  # Check if response variable is binary
  if (!is.factor(data[[response]]) || length(levels(data[[response]])) != 2) {
    stop("Response variable must be binary (factor with 2 levels)")
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

  # Fit Random Forest model
  rf_model <- randomForest(as.formula(paste(response, "~", paste(predictors, collapse = "+"))),
                           data = data_train)

  # Save the model
  saveRDS(rf_model, "random_forest_binary.rds")


  # Predict on training data
  train_predictions <- predict(rf_model, data_train)
  train_accuracy <- mean(train_predictions == data_train[[response]])
  cat("Accuracy on training data:", train_accuracy, "\n")

  # Predict on test data
  test_predictions <- predict(rf_model, data_test)
  test_accuracy <- mean(test_predictions == data_test[[response]])
  cat("Accuracy on test data:", test_accuracy, "\n")
  print("Random Forest model is saved in the present directory with the name random_forest_binary.rds")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")


  # Return the model and accuracies
  return(list(model = rf_model, accuracy_train = train_accuracy, accuracy_test = test_accuracy, predictors = predictors))
}