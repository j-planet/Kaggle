# ntree20_basicBenchmark.R
# Random forest benchmark using a 20-tree random forest for each data set.

library(randomForest)
set.seed(1279)
Nfiles = 15
for ( i in 1:Nfiles ) {
  trainFile <- paste("ACT", i, "_competition_training.csv", sep='')
  testFile <- paste("ACT", i, "_competition_test.csv", sep='')
  
  # using colClasses to speed up reading of files
  train <- read.csv(trainFile, header=TRUE, nrows=100)
  classes = sapply(train,class); 
  train <- read.csv(trainFile, header=TRUE, colClasses=classes)
  
  # data sets 1 and 6 are too large to fit into memory and run basic
  # random forest. Sample 20% of data set instead.
  if (i== 1 | i==6) {
    Nrows = length(train[,1])
    train <- train[sample(Nrows, as.integer(0.2*Nrows)),]
  }
  
  rf <- randomForest(train[,3:length(train)], train$Act, ntree=20, do.trace=2, mtry=25)
  test <- read.csv(testFile, header=TRUE)
  result <- predict(rf, test[,2:length(test)], type="response")
  
  if ( i == 1 ) {
    submission <- data.frame(test$MOLECULE, result)
    colnames(submission) <- c("MOLECULE", "Prediction")
  }
  else {
    nextSub <- data.frame(test$MOLECULE, result)
    colnames(nextSub) <- c("MOLECULE", "Prediction")
    submission <- rbind(submission, nextSub)
  }  
}

write.csv(submission, "ntree20_basicBenchmark.csv", quote=TRUE, row.names=FALSE)
