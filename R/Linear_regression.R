#' ----------------------------------------------------------------------------------------------------------------------------------
#' Fit Linear Regression Model
#' ----------------------------------------------------------------------------------------------------------------------------------
#' 
#' This function fits a linear regression model using the provided dataset and evaluates its accuracy.
#' 
#' @param data A dataframe containing the dataset.
#' @param predictor A character vector specifying the predictor variables.
#' @param response A character vector specifying the response variable.
#' @param top_k An integer indicating the number of top features to select for modeling (default is NULL).
#' 
#' @return A list containing the fitted linear regression model, R-squared on test data, test predictions, and selected predictors.
#' 
#' @details
#' The function performs the following steps:
#' - Removes missing values from the dataset.
#' - Checks if the input data is a dataframe and if the response variable exists in the dataframe.
#' - If 'top_k' is specified, selects the top 'k' features using the 'select_top_features' function.
#' - Splits the dataset into training and testing sets using the 'train_test_split' function.
#' - Fits a linear regression model using the 'lm' function.
#' - Saves the model as 'linear_regression.rds' in the present directory.
#' - Calculates R-squared on both the training and testing data.
#' 
#' @examples
#' fit_linear_regression(data = my_data, predictor = c("feature1", "feature2"), response = "target_variable", top_k = 5)
#' 
#' @importFrom stats lm predict summary
#' @importFrom utils saveRDS
#' @importFrom dplyr na.omit select
#' @importFrom tidyr gather
#' @importFrom caret trainTestSplit
#' @export
#' 
#' 
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
