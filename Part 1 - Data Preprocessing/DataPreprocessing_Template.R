#Data Processing

#Importing dataset
dataset = read.csv("Data.csv")
#dataset = dataset[,2:3]

#Take care of missing data
dataset$Age = ifelse(is.na(dataset$Age),
                ave(dataset$Age, FUN = function(x) mean(x,na.rm = TRUE)),
                dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x,na.rm = TRUE)),
                     dataset$Salary)

#Encoding categorical data
dataset$Country = factor(dataset$Country,
                         levels=c('France','Spain','Germany'),
                         labels=c(1,2,3))

dataset$Purchased = factor(dataset$Purchased,
                         levels=c('No','Yes'),
                         labels=c(0,1))

#Splitting data into training and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_Set = subset(dataset, split == TRUE)
test_Set = subset(dataset, split == FALSE)

#Feature Scaling
training_Set[,2:3] = scale(training_Set[,2:3])
test_Set[,2:3] = scale(test_Set[,2:3])