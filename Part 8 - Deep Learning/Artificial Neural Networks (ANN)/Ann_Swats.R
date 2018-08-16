#Artificial Neural Network

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

# Encoding the categorical data as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                  levels = c('Female', 'Male'),
                                  labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
#We need to apply feature scaling for training a neural network because it is computation intensive
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Fitting ANN to the Training set
#Neural - Deep learning models for regressors
#Nnet - Deep learning models with only one hidden layer
#DeepNet - Deep Learning models with many hidden layers
#H2O - Open source s/w platform, that allows connection to a instance of computer systems and allows to train the model efficiently
#      Gives lots of options to build a deep learning model
#     It contains parameter tuning arguments that helps to choose optimal numbers to build deep learning model
#install.packages('h2o')
library(h2o)
#Initialising the remote connection, nthreads - no. of cores in the system you want to connect to build deep learning model
# -1 - take all the cores/ optimize the no. of cores in the system
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = 'Exited',
                              training_frame = as.h2o(training_set),
                              activation = 'Rectifier',
                              hidden = c(6, 6),
                              epochs = 100,
                              train_samples_per_iteration = -2)

# Predicting the Test set results
prob_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
y_pred = (prob_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)

#Disconnect from the remote server
h2o.shutdown()