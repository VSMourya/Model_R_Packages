
# Function to fit Ridge Logistic Regression and evaluate accuracy

fit_logistic_ridge <- function(data, predictors, response, lambda = 1, top_k = NULL) {

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

  # Fit Ridge Logistic Regression model
  ridge_model <- glmnet(x_train, y_train, alpha = 0, family = "binomial", lambda = lambda)

  saveRDS(ridge_model, "ridge_logistic")


  # Predict and calculate accuracy for the training data
  predictions_train <- predict(ridge_model, s = lambda, newx = x_train, type = "response")
  correct_predictions_train <- mean((predictions_train > 0.5) == y_train)

  # Predict and calculate accuracy for the test data
  predictions_test <- predict(ridge_model, s = lambda, newx = x_test, type = "response")
  correct_predictions_test <- mean((predictions_test > 0.5) == y_test)

  cat("Accuracy on train data:", correct_predictions_train, "\n")
  cat("Accuracy on test data:", correct_predictions_test, "\n")
  print("Ridge Logistic Regression model is saved in the present directory with the name ridge_logistic")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")


  return(list(model = ridge_model, accuracy_train = correct_predictions_train, accuracy_test = correct_predictions_test, predictors = predictors))
}
