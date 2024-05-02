# This project is going to be a lazy predictor. 
# It is abled to be used both for Logistic Regression and Linear Regression. 
# Methods to be used are as follows:
# 1. Linear/Logistic Regression
# 2. Ridge Regression
# 3. Lasso Regression
# 4. Elastic Net
# 5. Support Vector Machine, Random Forest, Boosted Trees
# Implement Forward Selection.
# Perform Bagging - Sampling with Replacement - Write own function.
# Final model will be average from different models generated using Bagging. 
# Implement Ensemble model to fit more than one model - Write own code. 
# Naive weight for predictors - What is this ?


library(caret)
library(glmnet)

# Helper Functions (put each in a separate part of the script or keep them together if short)
source("preprocessing_helpers.R")  # Assuming you organize helpers in separate files

# Main Regression Function
perform_regressions <- function(file_path, target_col, regression_type = "linear", is_binary = FALSE) {
  # Load and prepare data
  df <- read.csv(file_path)
  df <- preprocess_data(df)  # Assume preprocessing includes all necessary steps
  
  # Choose the regression method
  if (regression_type == "linear") {
    model <- linear_regression(df, target_col)
  } else if (regression_type %in% c("ridge", "lasso", "elasticnet")) {
    model <- glmnet_regression(df, target_col, type = regression_type, is_binary = is_binary)
  }
  
  # Output model and summary
  list(model = model, summary = summary(model))
}

# Linear Regression
linear_regression <- function(df, target_col) {
  formula <- as.formula(paste(target_col, "~ ."))
  model <- lm(formula, data = df)
  model
}

# GLMNET for Ridge, Lasso, and Elastic Net
glmnet_regression <- function(df, target_col, type = "ridge", is_binary = FALSE) {
  x <- model.matrix(as.formula(paste(target_col, "~ .")), df)[,-1]  # Exclude intercept
  y <- df[[target_col]]
  
  if (is_binary) {
    family <- "binomial"
  } else {
    family <- "gaussian"
  }
  
  alpha <- ifelse(type == "ridge", 0, ifelse(type == "lasso", 1, 0.5))
  model <- glmnet(x, y, alpha = alpha, family = family)
  model
}