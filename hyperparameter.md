HYPERPARAMETER TUNING
================
Huizi Yu
9/24/2019

From the previous section, we found the largest Rsquared 53.1% is achieved when we remove 3000 data points using SOD outlier removal method (5 folds cross validation). For this section, we first remove said 3000 outliers and attempt to tune the hyperparameter in random forest.

Loading Concrete Data into workplace
------------------------------------

``` r
setwd("~/Hyperparameter")
library(glmnet)
```

    ## Loading required package: Matrix

    ## Loading required package: foreach

    ## Loaded glmnet 2.0-16

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## 
    ## Attaching package: 'ggplot2'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     margin

``` r
library(abodOutlier)
```

    ## Loading required package: cluster

``` r
library(standardize)
library(OutliersO3)
library(OutlierDetection)
library(neuralnet)
library(HighDimOut)
library(caret)
library(tree)
library(gbm)
```

    ## Loaded gbm 2.1.5

``` r
library(xgboost)
library(ranger)
```

    ## 
    ## Attaching package: 'ranger'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     importance

``` r
concrete <- read.csv("Clean_data.csv")
SOD <- read.csv("SOD.csv")
```

Removing the outliers
---------------------

``` r
input <- concrete[,1:8]
input2 <- scale(input)
complete <- cbind(input2, concrete$overdesign)
concrete2 <- as.data.frame(complete) 
colnames(concrete2) <- c("coarse_agg_weight", "fine_agg_weight", "current_weight", "fly_ash_weight", "AEA_dose", "type_awra_dose", "weight_ratio", "target", "overdesign")
concrete2[order(SOD, decreasing = TRUE)[1:3000],"Ind"] <- "Outlier"
concrete2[is.na(concrete2$Ind),"Ind"] <- "Inlier"
train_sov <- subset(concrete2, concrete2$Ind == "Inlier")
```

Setting benchmark before outlier removal (we also use five folds cross validation)
----------------------------------------------------------------------------------

``` r
Rsquared_rf_avg <- c(NA, NA, NA, NA, NA)

for (j in 1:5) {
  set.seed(1234567+j*1000) 
  samp<-sample(1:nrow(train_sov),nrow(train_sov)*0.8,replace = F)
  train <-train_sov[samp,]
  test <- train_sov[-samp,]
  mu <- mean(test$overdesign)
  tree_abod <- randomForest(y = train$overdesign , x = train[,1:8], ntree = 500, importance = TRUE)
  rf.pred_abod <- predict(tree_abod, newdata =as.matrix(test[,1:8]))
  Rsquared_abod_1 <- 1 - (sum((test$overdesign - rf.pred_abod)^2)/sum((test$overdesign - mu)^2))
  Rsquared_rf_avg[j] = Rsquared_abod_1
}

mean(Rsquared_rf_avg)
```

    ## [1] 0.5317221

Hyperparameter tuning using "ranger" function (due to its fast computational speed)
-----------------------------------------------------------------------------------

### (a) mtry: number of variables tried at individual tree

### (b) node\_size: maximum number of node

### (c) sample\_size: number of sample used to build tree

``` r
hyper_grid <- expand.grid(
  mtry       = seq(1, 8, by = 1),
  node_size  = seq(3, 9, by = 2),
  sampe_size = c(.55, .632, .70, .80),
  OOB_RMSE   = 0
)

for(i in 1:nrow(hyper_grid)) {
  # train model
  model <- ranger(
    formula         = overdesign~., 
    data            = train_sov, 
    num.trees       = 500,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sampe_size[i],
    seed            = 1234567
  )
  # add OOB error to grid
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

# best combination
ordered <- hyper_grid[order(hyper_grid$OOB_RMSE),]
ordered[1,]
```

    ##     mtry node_size sampe_size  OOB_RMSE
    ## 116    4         7        0.8 0.1436468

Testing the tuned random forest on testing data set using five fold cross validation
------------------------------------------------------------------------------------

``` r
Rsquared_rf_avg <- c(NA, NA, NA, NA, NA)

for (j in 1:5) {
  set.seed(1234567+j*1000) 
  samp<-sample(1:nrow(train_sov),nrow(train_sov)*0.8,replace = F)
  train <-train_sov[samp,]
  test <- train_sov[-samp,]
  mu <- mean(test$overdesign)
  tree_abod <- ranger(
    formula         = overdesign~., 
    data            = train_sov, 
    num.trees       = 500,
    mtry            = 4,
    min.node.size   = 7,
    sample.fraction = 0.8,
    seed            = 1234567
  )
  
  rf.pred_abod <- predict(tree_abod, data = test)
  Rsquared_abod_1 <- 1 - (sum((test$overdesign - rf.pred_abod$predictions)^2)/sum((test$overdesign - mu)^2))
  Rsquared_rf_avg[j] = Rsquared_abod_1
}

mean(Rsquared_rf_avg)
```

    ## [1] 0.8335155

By tuning the hyperparameters, we achieve the new Rsquared of 83.35 %, almost 30 % improvement from the benchmark.
------------------------------------------------------------------------------------------------------------------
