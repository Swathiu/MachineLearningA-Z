#Regression Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

#Fitting the Regression Model to the dataset
#Adding Polynomial Features

#Predicting new result with Polynomial Regression
y_pred_poly = predict(regressor, data.frame(Level = 6.5))

#Visualising Polynomial Regression Results (for high resolution and smoother curve)
#library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor,newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Position Level') +
  ylab('Salary')