
# Function to fit Boosted Trees Linear Regression and evaluate accuracy

fit_boosted_trees <- function(data, predictors, response, top_k = NULL) {

  if (!require(gbm, quietly = TRUE)) {
    install.packages("gbm")
    library(gbm)
  } else {
    cat("Package 'gbm' is already installed.\n")
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

  # Fit Boosted Trees model
  boosted_model <- gbm(as.formula(paste(response, "~", paste(predictors, collapse = "+"))),
                       data = data_train,
                       distribution = "gaussian",
                       n.trees = 100,
                       interaction.depth = 4,
                       shrinkage = 0.01,
                       bag.fraction = 0.5,
                       cv.folds = 5)

  # Save the model
  saveRDS(boosted_model, "boosted_trees.rds")

  # Predict on training data
  train_predictions <- predict(boosted_model, data_train, n.trees = 100)
  train_rmse <- sqrt(mean((train_predictions - data_train[[response]])^2))
  cat("RMSE on training data:", train_rmse, "\n")

  # Predict on test data
  test_predictions <- predict(boosted_model, data_test, n.trees = 100)
  test_rmse <- sqrt(mean((test_predictions - data_test[[response]])^2))
  cat("RMSE on test data:", test_rmse, "\n")
  print("Boosted Trees model is saved in the present directory with the name boosted_trees.rds")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")

  # Return the model and RMSE values
  return(list(model = boosted_model, rmse_train = train_rmse, rmse_test = test_rmse, predictors = predictors))
}
