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

#Fitting Linear Regression to the dataset (Only for the sake of comparision of the results)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
#poly_reg object will be transformer tool that will transform our matrix of features X into
#a new matrix of features that will be called X_poly which will contain powers  (2,3,4,...n) of independent variables
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

polylin_reg = LinearRegression()
polylin_reg.fit(X_poly,Y)

#Visualising linear regression model results
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising polynomial regression results
X_Grid = np.arange(min(X),max(X),0.1)
X_Grid = X_Grid.reshape((len(X_Grid)),1)
plt.scatter(X,Y,color='red')
plt.plot(X_Grid,polylin_reg.predict(poly_reg.fit_transform(X_Grid)),color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predict new result with Linear Regression
lin_reg.predict(6.5)

#Predict new result with Polynomial Regression
polylin_reg.predict(poly_reg.fit_transform(6.5))






