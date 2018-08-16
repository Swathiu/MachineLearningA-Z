
#Simple Linear Regression

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Loading the dataset
dataset = pd.read_csv("Salary_Data.csv")

#Create a matrix of independant variables
X = dataset.iloc[:,:-1].values

#Create a vector of dependant variables
Y = dataset.iloc[:,1].values


#Split the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

#IN SLR, the algorithm will take care of scaling the training set

#Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting Test Set results
y_pred = regressor.predict(X_test)

#Visualizing Test Set results and training set
plt.scatter(X_train, Y_train,color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test,Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
