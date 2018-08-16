#Theano and Tensorflow used to build deep learning networks from scratch. Mainly used in research work.
#Keras wraps the above two libraries. It is built on top of above libraries

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encode Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])
labelencoder_X_gender = LabelEncoder()
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#Splitting dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#You need to apply feature scaling to ease all the calculations
#Feature Scaling - We use feature scaling so large variable set don't drown out smaller ones
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building ANN
#Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing ANN
classifier = Sequential()

#Adding input layer and an hidden layer
#Dense function - Activation function, Number of inputs in the input layer, no. of nodes to be added in hidden layer
#Intializing weights according to uniform distribution
#Rectifier activation func for hidden layer and sigmoid activation func for output layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

#Adding second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#Adding the output layer, binary outcomes have only one output layer, sigmoid function is the heart of activation function
#If we have more than 2 classes - 3 or more, then we use softmax function
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Applying stochastic gradient to the neural network model - this process is called compiling
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

#Predicting Test Set Results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
