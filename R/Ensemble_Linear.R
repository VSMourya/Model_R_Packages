#' ---------------------------------------------------------------------------------------------------------------
#' Ensemble Model for Linear Regression
#' ---------------------------------------------------------------------------------------------------------------
#' 
#' This function implements ensemble learning for linear regression by combining predictions from multiple regression models.
#'
#' @param data A dataframe containing the dataset.
#' @param predictors A character vector specifying the predictor variables.
#' @param response A character string specifying the response variable.
#' @param models A character vector specifying the types of regression models to include in the ensemble.
#'
#' @return A numeric vector containing the combined predictions from the ensemble model.
#'
#' @examples
#' data <- read.csv("data.csv")
#' predictors <- c("predictor1", "predictor2", "predictor3")
#' response <- "response"
#' models <- c("linear_regression", "ridge_regression", "lasso_regression")
#' ensemble_learning(data, predictors, response, models)
#'
#' @importFrom stats predict
#' @importFrom stats mean
#' @importFrom base print
#' @export

ensemble_learning <- function(data, predictors, response, models) {
  predictions <- list()
  for (model_type in models) {
    if (model_type == "linear_regression") {
      lin_reg <- fit_linear_regression(data, predictors, response)
      predictions[[model_type]] <- lin_reg$test_predictions
    } else if (model_type == "ridge_regression") {
      ridge_reg <- fit_linear_ridge_regression(data, predictors, response)
      predictions[[model_type]] <- predict(ridge_reg$model, newx = as.matrix(data[predictors]))
    } else if (model_type == "lasso_regression") {
      lasso_reg <- fit_linear_lasso_regression(data, predictors, response)
      predictions[[model_type]] <- predict(lasso_reg$model, newx = as.matrix(data[predictors]))
    } else if (model_type == "elastic_net") {
      elastic_reg <- fit_linear_elastic_net(data, predictors, response)
      predictions[[model_type]] <- predict(elastic_reg$model, newx = as.matrix(data[predictors]))
    } else if (model_type == "svm_regression") {
      svm_reg <- fit_linear_svm(data, predictors, response)
      predictions[[model_type]] <- predict(svm_reg$model, newdata = data)
    } else if (model_type == "random_forest") {
      rf_reg <- fit_random_forest(data, predictors, response)
      predictions[[model_type]] <- predict(rf_reg$model, newdata = data)
    } else if (model_type == "boosted_trees") {
      bt_reg <- fit_boosted_trees(data, predictors, response)
      predictions[[model_type]] <- predict(bt_reg$model, newdata = data, n.trees = 100)
    } else {
      print(paste("Model type", model_type, "not recognized"))
    }
  }

  combined_predictions <- rowMeans(do.call(cbind, predictions))

  squared_errors <- (data[[response]] - combined_predictions)^2
  mean_squared_error <- mean(squared_errors)
  rmse <- sqrt(mean_squared_error)
  cat("RMSE for emsemble model is:", rmse, "\n")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")
  return(combined_predictions)
}
