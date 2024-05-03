
# Function to fit Random Forest Linear Regression and evaluate accuracy

fit_random_forest <- function(data, predictors, response, top_k = NULL) {

  if (!require(randomForest, quietly = TRUE)) {
    install.packages("randomForest")
    library(randomForest)
  } else {
    cat("Package 'randomForest' is already installed.\n")
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

  # Fit Random Forest model
  rf_model <- randomForest(as.formula(paste(response, "~", paste(predictors, collapse = "+"))),
                           data = data_train)

  # Save the model
  saveRDS(rf_model, "random_forest.rds")

  # Predict on training data
  train_predictions <- predict(rf_model, data_train)
  train_rmse <- sqrt(mean((train_predictions - data_train[[response]])^2))
  cat("RMSE on training data:", train_rmse, "\n")

  # Predict on test data
  test_predictions <- predict(rf_model, data_test)
  test_rmse <- sqrt(mean((test_predictions - data_test[[response]])^2))
  cat("RMSE on test data:", test_rmse, "\n")
  print("Random Forest model is saved in the present directory with the name random_forest.rds")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")

  # Return the model and RMSE values
  return(list(model = rf_model, rmse_train = train_rmse, rmse_test = test_rmse, predictors = predictors))
}
