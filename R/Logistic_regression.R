
# Function to fit Logistic Regression and evaluate accuracy

fit_logistic_regression <- function(data, predictor, response, top_k = NULL) {

  data <- na.omit(data)
  if (!is.data.frame(data)) {
    stop("Data must be a dataframe")
  }
  if (!(response %in% names(data))) {
    stop("Predictor or response not found in the dataframe")
  }

  if (!is.null(top_k)) {
    predictor <- select_top_features(data = data, predictors = predictor , response = response, k = top_k)
    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    print("Selected top K Features")
    print(predictor)
  }

  split_result <- train_test_split(data)
  data_train <- split_result$train
  data_test <- split_result$test

  cat("Dimensions of Training Data - Rows:", nrow(data_train), "Columns:", ncol(data_train), "\n")
  cat("Dimensions of Testing Data - Rows:", nrow(data_test), "Columns:", ncol(data_test), "\n")

  print("Starting building model")
  log_model <- glm(reformulate(predictor, response), data = data_train, family = binomial())

  saveRDS(log_model, "logistic_regression.rds")

  # Calculate accuracy on training data
  predicted_probabilities_train <- predict(log_model, data_train, type = "response")
  predicted_classes_train <- ifelse(predicted_probabilities_train > 0.5, 1, 0)
  correct_predictions_train <- mean(predicted_classes_train == data_train[[response]])

  # Calculate accuracy on test data
  predicted_probabilities_test <- predict(log_model, data_test, type = "response")
  predicted_classes_test <- ifelse(predicted_probabilities_test > 0.5, 1, 0)
  correct_predictions_test <- mean(predicted_classes_test == data_test[[response]])

  print(log_model)
  print(summary(log_model))
  cat("Accuracy on train data:", correct_predictions_train, "\n")
  cat("Accuracy on test data:", correct_predictions_test, "\n")
  print("Model is saved in present directory with the name logistic_regression")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")


  return(list(model = log_model, accuracy_test = correct_predictions_test, test_predictions = predicted_classes_test, predictors = predictor))
}
