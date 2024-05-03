# Ensemble Model for Logistic Regression
ensemble_learning_binary <- function(data, predictors, response, models) {
  predictions <- list()

  # Fit models and collect predictions
  for (model_type in models) {
    if (model_type == "logistic_regression") {
      log_reg <- fit_logistic_regression(data, predictors, response)
      predictions[[model_type]] <- log_reg$test_predictions
    } else if (model_type == "ridge_regression") {
      ridge_reg <- fit_logistic_ridge(data, predictors, response)
      predictions[[model_type]] <- predict(ridge_reg$model, s = 1, newx = as.matrix(data[predictors]), type = "response")
    } else if (model_type == "lasso_regression") {
      lasso_reg <- fit_logistic_lasso(data, predictors, response)
      predictions[[model_type]] <- predict(lasso_reg$model, s = 1, newx = as.matrix(data[predictors]), type = "response")
    } else if (model_type == "elastic_net") {
      elastic_reg <- fit_logistic_elastic_net(data, predictors, response)
      predictions[[model_type]] <- predict(elastic_reg$model, s = 1, newx = as.matrix(data[predictors]), type = "response")
    } else if (model_type == "svm_regression") {
      svm_reg <- fit_logistic_svm(data, predictors, response)
      predictions[[model_type]] <- predict(svm_reg$model, newdata = data)
    } else if (model_type == "random_forest") {
      rf_reg <- fit_random_forest_binary(data, predictors, response)
      predictions[[model_type]] <- predict(rf_reg$model, newdata = data, type = "response")
    } else if (model_type == "boosted_trees") {
      bt_reg <- fit_boosted_trees_binary(data, predictors, response)
      predictions[[model_type]] <- predict(bt_reg$model, newdata = data, n.trees = 100, type = "response")
    } else {
      print(paste("Model type", model_type, "not recognized"))
    }
  }

  combined_predictions <- rowMeans(do.call(cbind, predictions))
  combined_classes <- ifelse(combined_predictions > 0.5, 1, 0)
  combined_accuracy <- mean(combined_classes == data[[response]])
  cat("Accuracy for emsemble model is:", combined_accuracy, "\n")
  print("--------------------------------------------------------------------------------------------------------------------------------------------")

  return(combined_classes)
}
