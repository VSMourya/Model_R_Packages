
# Function to fit SVM Logistic Regression and evaluate accuracy

fit_logistic_svm <- function(data, predictors, response, top_k = NULL) {

  if (!require(e1071, quietly = TRUE)) {
    install.packages("e1071")
    library(e1071)
  } else {
    cat("Package 'e1071' is already installed.\n")
  }

  # Assuming data_processed_log is your dataframe
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

  # Fit SVM model
  svm_model <- svm(as.formula(paste(response, "~", paste(predictors, collapse = "+"))),
                   data = data_train,
                   kernel = "radial")

  # Save the model
  saveRDS(svm_model, "logistic_svm.rds")


  # Predict on training data
  train_predictions <- predict(svm_model, data_train)
  train_accuracy <- mean(train_predictions == data_train[[response]])
  cat("Accuracy on training data:", train_accuracy, "\n")

  # Predict on test data
  test_predictions <- predict(svm_model, data_test)
  test_accuracy <- mean(test_predictions == data_test[[response]])
  cat("Accuracy on test data:", test_accuracy, "\n")
  print("SVM model is saved in the present directory with the name logistic_svm.rds")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")


  # Return the model and accuracies
  return(list(model = svm_model, accuracy_train = train_accuracy, accuracy_test = test_accuracy, predictors = predictors))
}
