
# Ensemble Model for Linear Regression

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
