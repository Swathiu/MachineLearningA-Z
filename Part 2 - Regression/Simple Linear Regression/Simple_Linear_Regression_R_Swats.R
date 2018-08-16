#Simple Linear Regression

#Importing dataset
dataset = read.csv("Salary_Data.csv")

#Splitting dataset into training and testset
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary,SplitRatio = 2/3)
training_set = subset(dataset,split==TRUE)
test_set = subset(dataset,split==FALSE)

#LinearRegression Algorithm will take care of Feature Scaling, so we need
#not explicit do it here

#Fitting Simple Linear Regression to Training Set
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

#Predict test set results
#Vector of Predictions = y_pred
y_pred = predict(regressor,test_set)

#Visualizing Training_Set and Test_Set results
#install.packages("ggplot2")
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience,y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor,newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training Set)') +
  xlab('Years of Experience') +
  ylab('Salary')

ggplot() +
  geom_point(aes(x = test_set$YearsExperience,y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor,newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training Set)') +
  xlab('Years of Experience') +
  ylab('Salary')

