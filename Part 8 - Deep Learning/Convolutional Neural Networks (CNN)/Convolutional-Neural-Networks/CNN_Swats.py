#Convolutional Neural Network

#Importing libraries

#Sequential Package used to initialize our neural networks. There are 2 ways to initialize a neural network either as a sequence of layers or as a graph
from keras.models import Sequential
#Used to make first step of CNN - To add convolutional layers and also images are 2D unlike videos that are 3D.
from keras.layers import Convolution2D
#It will add pooling layers
from keras.layers import MaxPooling2D
#Converts all pooling feature layers to large feature vectors
from keras.layers import Flatten
#Add fully connected layers in a classic Artificial Neural Network
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Step -1 Convolution Layer
#Add how many feature detectors we need to add into the convolutional layer thereby adding number of feature maps in Convolution Layer
#3 - 3 dimensions for colored image - RGB (No. of channels)
#We add activation func - to get non-linearity and remove negative values
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Step -2 Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step -3 Flattening
classifier.add(Flatten())

#Step -4 Full Connection
#Adding hidden layer/ fully connected layer
classifier.add(Dense(output_dim = 128, activation = 'relu'))
#Adding output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Compiling CNN - Modifying the weights, feature detectors to optimize the performance of the model by using stochastic gradient descent function (loss metric) and a performance metric
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting CNN to images
#Flow from Directory method from Image Processing in Keras Documentation

#Adding one more convolution layer and a fully connected layer will increase the accuracy of test set and prevents overfitting of training set

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')    #Binary Outcome

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)