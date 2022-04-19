library(knitr)
library(caret)
library(rpart)
library(corrplot)
library(RColorBrewer)
library(gbm)
library(rpart)

#Download data
#download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = 'pml-training.csv')
#download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = 'pml-testing.csv')

raw_train <- read.csv("pml-training.csv")
raw_test <- read.csv("pml-testing.csv")

inTrain  <- createDataPartition(raw_train$classe, p=0.7, list=FALSE)

train_set <- raw_train[inTrain, ]

test_set  <- raw_train[-inTrain, ]
dim(train_set)

# Indicating columns with the near zero variance
nzv <- nearZeroVar(train_set)

#Exclude nzv columns from datasets
train_set <- train_set[, -nzv]
test_set <- test_set[, -nzv]

dim(train_set)

# 56 columns have been excluded since we suppose that zero variance data has no impact to the model
# Then we remove the rows which are mostly NA
mostlyNA <- sapply(train_set, function(x){ 
  mean(is.na(x)) > .95
  })
train_set <- train_set[, mostlyNA == FALSE]
test_set <- test_set[, mostlyNA == FALSE]
dim(train_set)

# Only 59 columns left that have impact to the model
train_set <- train_set[, -(1:5)]
test_set <- test_set[, -(1:5)]
names(train_set)


# Now we gonna find correlations between variables

corMatrix <- cor(train_set[, -54])
corrplot(corMatrix, order = "hclust", method = "color", type = "upper", 
         col=brewer.pal(n=8, name="RdYlBu"))

# We gonna compare three types of predictions and select the best of them

# 1. Random forest
set.seed(111)
control_forest <- trainControl(method="cv", number=3, verboseIter=FALSE)
modeled_fit_rf <- train(classe ~ ., data=train_set, method="rf",
                          trControl=control_forest)
modeled_fit_rf$finalModel
predict_rf <- predict(modeled_fit_rf, newdata=test_set)
conf_rf <- confusionMatrix(predict_rf, as.factor(test_set$classe))
conf_rf

# 2. General boosted model
# Fit the model
control_gbm <- trainControl(method = "repeatedcv", number = 4, repeats = 2)
fit_gbm  <- train(classe ~ ., data=train_set, method = "gbm",
                    trControl = control_gbm, verbose = FALSE)
fit_gbm$finalModel
predict_gbm <- predict(fit_gbm, newdata = test_set)
conf_gbm <- confusionMatrix(predict_gbm, as.factor(test_set$classe))
conf_gbm

# As we can see the Random forest model has better accuracy (0.9968 against 0.9884) so we gonna use it to predict test data
predict_final <- predict(modeled_fit_rf, newdata=raw_test)
predict_final
