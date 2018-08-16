#Regression template


#Polynomial Regression

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Data Preprocessing
#Loading dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values #Adding 1:2 instead of 1, so that X is not considered as array and considered as matrix
Y = dataset.iloc[:,2].values

#We will not split data into test set and training set owing to the very small size of the dataset, it would not make 
#much sense to split the data. Also, we want to make accurate prediction here. 

#Fitting the Regression Model to the dataset and creating a regressor.

#Predict new result with Polynomial Regression
y_pred = regressor.predict(6.5)

#Visualising polynomial regression results
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#Visualising Regression results with higher resolution
X_grid = np.arange(min(X),max(X),0.1)
X_grid = np.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


