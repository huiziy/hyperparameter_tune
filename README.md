HYPERPARAMETER TUNING
================
Huizi Yu
9/24/2019

From the previous section, we found the largest Rsquared 51.167% is achieved when we remove 2450 data points using SOD outlier removal method (5 folds cross validation). Removing more data points will lead to fluctuation around the 50% mark. For this section, we first remove said 2450 outliers and attempt to tune the hyperparameter in random forest. More specifically, we attempt to tune the following four parameters: number of trees, mtry (number of variables available for splitting at each tree node), maximum number of nodes, and sample fraction used to grow the forest (the rest is used to calculate Out-Of-Bag Error).

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
concrete2[order(SOD, decreasing = TRUE)[1:2450],"Ind"] <- "Outlier"
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
  tree_abod <- randomForest(y = train$overdesign , x = train[,1:7], ntree = 500, importance = TRUE)
  rf.pred_abod <- predict(tree_abod, newdata =as.matrix(test[,1:7]))
  Rsquared_abod_1 <- 1 - (sum((test$overdesign - rf.pred_abod)^2)/sum((test$overdesign - mu)^2))
  Rsquared_rf_avg[j] = Rsquared_abod_1
}

mean(Rsquared_rf_avg)
```

    ## [1] 0.5116779

Hyperparameter tuning using "ranger" function (due to its fast computational speed)
-----------------------------------------------------------------------------------

### (a) mtry: number of variables tried at individual tree

### (b) num\_trees: the number of trees in each random forest

### (b) node\_size: maximum number of node

### (c) sample\_size: number of sample used to build tree

``` r
hyper_grid <- expand.grid(
  num_trees  = c(300,500,800,1000),
  mtry       = seq(1, 7, by = 1),
  node_size  = seq(3, 9, by = 2),
  sampe_size = c(.55, .632, .70, .80),
  OOB_RMSE   = 0
)

for(i in 1:nrow(hyper_grid)) {
  # train model
  model <- ranger(
    formula         = overdesign~coarse_agg_weight+fine_agg_weight+current_weight+fly_ash_weight+AEA_dose+type_awra_dose+weight_ratio, 
    data            = train_sov, 
    num.trees       = hyper_grid$num_trees[i],
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

    ##     num_trees mtry node_size sampe_size  OOB_RMSE
    ## 152      1000    3         5      0.632 0.1481207

Testing the tuned random forest on testing data set using five fold cross validation
------------------------------------------------------------------------------------

``` r
Rsquared_rf_avg_tuned <- c(NA, NA, NA, NA, NA)

for (j in 1:5) {
  set.seed(1234567+j*1000) 
  samp<-sample(1:nrow(train_sov),nrow(train_sov)*0.8,replace = F)
  train <-train_sov[samp,]
  test <- train_sov[-samp,]
  test_x <- test[,1:7]
  mu <- mean(test$overdesign)
  tree_abod <- ranger(
    formula         = overdesign~coarse_agg_weight+fine_agg_weight+current_weight+fly_ash_weight+AEA_dose+type_awra_dose+weight_ratio, 
    data            = train, 
    num.trees       = 1000,
    mtry            = 3,
    min.node.size   = 5,
    sample.fraction = 0.632,
    seed            = 1234567
  )
  
  rf.pred_abod <- predict(tree_abod, data = test_x)
  Rsquared_abod_1 <- 1 - (sum((test$overdesign - rf.pred_abod$predictions)^2)/sum((test$overdesign - mu)^2))
  Rsquared_rf_avg_tuned[j] = Rsquared_abod_1
}

mean(Rsquared_rf_avg_tuned)
```

    ## [1] 0.5128897

By tuning the hyperparameters, we achieve the new Rsquared of 51.28897%, which is a very minor improvement comparing to the benchmark 51.16779% (after outlier removal).
------------------------------------------------------------------------------------------------------------------------------------------------------------------------

