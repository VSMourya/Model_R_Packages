
# Function to fit Linear Regression and evaluate accuracy


fit_linear_regression <- function(data, predictor, response, top_k = NULL) {

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
  #df_op <- data_train[, which(names(data_train) == response)]
  #df_inp <- data_train[, -which(names(data_train) == response)]
  #predictors <- names(df_inp)
  #formula <- as.formula(paste(response, "~", paste(predictors, collapse = "+")))
  #model <- lm(formula, data = data_train)

  lin_model <- lm(reformulate(predictor, response), data = data)

  saveRDS(lin_model, "linear_regression.rds")

  r_squared_train <- summary(lin_model)$r.squared

  predictions <- predict(lin_model, data_test)

  ss_total <- sum((data_test[[response]] - mean(data_test[[response]]))^2)
  ss_res <- sum((data_test[[response]] - predictions)^2)
  r_squared <- 1 - (ss_res / ss_total)

  print(lin_model)
  print(summary(lin_model))

  cat("R-squared on train data:", r_squared_train, "\n")
  cat("R-squared on test data:", r_squared, "\n")
  print("Model is saved in present directory with the name linear_regression.rds")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")

  return(list(model = lin_model, r_squared = r_squared,test_predictions = predictions, predictors = predictor))
}
