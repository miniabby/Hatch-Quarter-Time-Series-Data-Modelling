library(tidyverse)
library(tsibble)
library(fable)
library(fable.prophet)
library(feasts)
library(readxl)

setwd("/hdd/Dropbox/Data Science Project/Combined Datasets For AUS and US")
data <- read_csv("08-19_combined.csv") %>% mutate(month = yearmonth(as.Date(month, format = "%m/%d/%Y"))) %>% 
  as_tsibble() %>% rename(month = month)

# baseline forecasts for 2019 Jul - Dec and 2019 Jan - Dec
base_6m <- data %>% filter(month < yearmonth("2019 Jul")) %>% 
  model(naive = NAIVE(import_ratio),
        snaive = SNAIVE(import_ratio),
        mean = MEAN(import_ratio))
base_12m <- data %>% filter(month < yearmonth("2019 Jan")) %>% 
  model(naive = NAIVE(import_ratio),
        snaive = SNAIVE(import_ratio),
        mean = MEAN(import_ratio))

base_6m_fc <- base_6m %>% forecast(h = 6)
base_12m_fc <- base_12m %>% forecast(h = 12)

accuracy(base_6m_fc, data)
accuracy(base_12m_fc, data)

# function for predicting with prophet
# data - whole dataset
# horizon - forecast horizon
# cutoff - first month to be forecasted (as str)
# xvar - vector of feature names
fc_prophet <- function(data, cutoff, xvar, option = 1) {
  train <- data %>% filter(month < yearmonth(cutoff)) %>% 
    select(xvar, import_ratio)
  test <- data %>% filter(month >= yearmonth(cutoff), month < yearmonth("2020 Jan")) %>% 
    select(xvar, import_ratio)
  
  xvar <- sapply(xvar, function (x) {
    x <- gsub('[^A-Za-z0-9_ ]', '', x)
    gsub(' ', '_', x)
  }) %>% as.vector()
  
  colnames(train) <- c(xvar, "import_ratio", "month")
  colnames(test) <- colnames(train)
  xvar <- xvar %>% paste(collapse = "+")
  
  fit <- train %>% model(prophet(import_ratio ~ . -import_ratio -month + season(period = 12, order = 10, type = "multiplicative")))
  fc <- fit %>% forecast(new_data = test)
  
  if (option == 1) {
    return(fc)
  } else if (option == 2) {
    return(list(fit, test))
  } else {
    return(fit %>% forecast(new_data = train))
  }
}

# prophet with no features
pr_6m <- data %>% filter(month < yearmonth("2019 Jul")) %>% 
  model(prophet(import_ratio ~ season(period = 12, order= 10, type = "multiplicative")))
pr_12m <- data %>% filter(month < yearmonth("2019 Jan")) %>% 
  model(prophet(import_ratio ~ season(period = 12, order= 10, type = "multiplicative")))

pr_6m_fc <- pr_6m %>% forecast(h = 6)
pr_12m_fc <- pr_12m %>% forecast(h = 12)

accuracy(pr_6m_fc, data)
accuracy(pr_12m_fc, data)

# prophet with shap selected features
setwd("~/Documents/Uni/Master/2021/Project/Data-Science-Project/XGBoost")
shap_x6 <- read_csv("updated_6m_shap_rank.csv") %>% select(-1) %>% filter(importance > 0)
shap_x12 <- read_csv("updated_12m_shap_rank.csv") %>% select(-1) %>% filter(importance > 0)

shap6 <- fc_prophet(data, "2019 Jul", shap_x6$features)
shap12 <- fc_prophet(data, "2019 Jan", shap_x12$features)

accuracy(shap6, data)
accuracy(shap12, data)

# prophet with xgboost selected features
xgb_x6 <- read_csv("updated_6m_xgboost_importance_rank.csv") %>% select(-1) %>% 
  filter(`0` > 0)
xgb_x12 <- read_csv("updated_12m_xgboost_importance_rank.csv") %>% select(-1) %>% 
  filter(`0` > 0)

xgb6 <- fc_prophet(data, "2019 Jul", xgb_x6$`1`)
xgb12 <- fc_prophet(data, "2019 Jan", xgb_x12$`1`)

accuracy(xgb6, data)
accuracy(xgb12, data)

# validate on all data for 2014 apr to oct (takes a while to run)
val <- fc_prophet(data, "2020 Jan", xgb_x6$`1`[1:32], 3)

val %>% filter(month >= yearmonth("2014 Apr"), month <= yearmonth("2014 Sep")) %>% 
  accuracy(data) -> val_acc


# backwards stepwise selection function
stepwise <- function(data, cutoff, xvar, threshold) {
  train <- data %>% filter(month < yearmonth(cutoff)) %>% 
    select(xvar, import_ratio)
  test <- data %>% filter(month >= yearmonth(cutoff), month < yearmonth("2020 Jan")) %>% 
    select(xvar, import_ratio)
  
  xvar <- sapply(xvar, function (x) {
    x <- gsub('[^A-Za-z0-9_ ]', '', x)
    gsub(' ', '_', x)
  }) %>% as.vector()
  
  colnames(train) <- c(xvar, "import_ratio", "month")
  colnames(test) <- colnames(train)
  
  fit <- train %>% model(prophet(import_ratio ~ .-import_ratio -month + season(period = 12, order = 10, type = "multiplicative")))
  fc <- fit %>% forecast(new_data = test)
  acc <- accuracy(fc, data)
  
  # backwards stepwise selection algorithm
  stop <- F
  counter <- length(xvar)
  
  # iterate until model does not improve
  while (!stop) {
    train <- train %>% select(-xvar[counter])
    temp_fit <- train %>% model(prophet(import_ratio ~ . -import_ratio -month + season(period = 12, order = 10, type = "multiplicative")))
    temp_fc <- temp_fit %>% forecast(new_data = test)
    temp_acc <- accuracy(temp_fc, data)
    
    if (temp_acc$RMSE/acc$RMSE < threshold) {
      acc <- temp_acc
      counter <- counter - 1
    } else {
      stop <- T
      return(list(counter, acc))
    }
  }
}

# perform backwards step selection
shap_step6 <- stepwise(data, "2019 Jul", shap_x6$features, 1.5)
xgb_step6 <- stepwise(data, "2019 Jul", xgb_x6$`1`, 5)

shap_step12 <- stepwise(data, "2019 Jan", shap_x12$features, 1)
xgb_step12 <- stepwise(data, "2019 Jan", xgb_x12$`1`, 1.5)



# feature selection according to ranking
feat_rank <- function(data, cutoff, xvar) {
  scores <- rep(0, length(xvar))
  train <- data %>% filter(month < yearmonth(cutoff)) %>% 
    select(xvar, import_ratio)
  test <- data %>% filter(month >= yearmonth(cutoff), month < yearmonth("2020 Jan")) %>% 
    select(xvar, import_ratio)
  
  xvar <- sapply(xvar, function (x) {
    x <- gsub('[^A-Za-z0-9_ ]', '', x)
    gsub(' ', '_', x)
  }) %>% as.vector()
  
  colnames(train) <- c(xvar, "import_ratio", "month")
  colnames(test) <- colnames(train)
  
  for (i in 1:length(xvar)) {
    temp_train <- train %>% select(1:i, import_ratio, month)
    fit <- temp_train %>% model(prophet(import_ratio ~ .-import_ratio -month + season(period = 12, order = 10, type = "multiplicative")))
    fc <- fit %>% forecast(new_data = test)
    scores[i] <- accuracy(fc, data)$RMSE
  }
  return(scores)
}

shap_rank6 <- feat_rank(data, "2019 Jul", shap_x6$features)
xgb_rank6 <- feat_rank(data, "2019 Jul", xgb_x6$`1`)

shap_rank12 <- feat_rank(data, "2019 Jan", shap_x12$features)
xgb_rank12 <- feat_rank(data, "2019 Jan", xgb_x12$`1`)

# models selected from features for 6 months
shap_29 <- fc_prophet(data, "2019 Jul", shap_x6$features[1:29])
xgb_32 <- fc_prophet(data, "2019 Jul", xgb_x6$`1`[1:32])
  
# models selected from features for 12 months
xgb_26 <- fc_prophet(data, "2019 Jan", xgb_x12$`1`[1:26])

# graphs for final models
final6 <- fc_prophet(data, "2019 Jul", xgb_x6$`1`[1:32])
final12 <- fc_prophet(data, "2019 Jan", xgb_x12$`1`[1:26])

final6 %>% autoplot() + geom_line(aes(x = month, y = import_ratio), data = data %>% filter(month > yearmonth("2017 Dec"))) +
  xlab("Time") + ylab("Import ratio") + ggtitle("Prophet 6 month forecast with 17 variables")

final12 %>% autoplot() + geom_line(aes(x = month, y = import_ratio), data = data %>% filter(month > yearmonth("2017 Dec"))) +
  xlab("Time") + ylab("Import ratio") + ggtitle("Prophet 12 month forecast with 10 variables")

decomp6 <-fc_prophet(data, "2019 Jul", xgb_x6$`1`[1:32], 2)
decomp6[[1]] %>% components() %>% autoplot()

################################################################################
# models using final selected features from xgb
shap6m <- c('us_livestock_commercial_lambs and yearlings', 'aus_beef&veal_exports', 
 'us_livestock_commercial_beef', 'aus_total dairy cattle and calves', 
 'aus_Meat Produced ;  CATTLE (excl. calves) ;  Total (State) Original (tonnes)', 
 'us_veganism', 'aus_avg_rainfall', 'us_veganism_trend_ratio', 
 'aus_Meat Produced ;  CATTLE (excl. calves) ;  Total (State) Seasonally adjusted (tonnes)', 
 'us cattle numbers(*1000)', 'us_fed_avg_dressed_calves',
 'aus_exports of beef, veal and live cattle to US',
 'us_production_fedral_inspected_total poultry', 'us_cold_storage_beef', 
 'us_production_commercial_beef', 'us_livestock_commercial_mature sheep', 
 'us_livestock_commercial_steers')

shap12m <- c('aus_beef&veal_slaughterings', 'aus_population_natural_increase', 
            'aus_beef&veal_exports', 'us_production_fedral_inspected_total red meat and poultry',
            'aus_exports of beef, veal and live cattle to US', 
            'us_livestock_commercial_mature sheep', 'us_livestock_commercial_beef',
            'us_veganism_trend_ratio', 'us_fed_avg_dressed_calves','us_veganism')

shap6m_fc <- fc_prophet(data, "2019 Jul", shap6m)
shap12m_fc <- fc_prophet(data, "2019 Jan", shap12m)

accuracy(shap6m_fc, data)
accuracy(shap12m_fc, data)
