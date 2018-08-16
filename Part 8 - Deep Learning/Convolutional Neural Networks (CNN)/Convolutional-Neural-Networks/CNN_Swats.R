
library(keras)
install_keras()

library(keras)
# Model
model <- keras_model_sequential()
model %>% 
  layer_conv_2d(filters = 32, kernel_size=c(3,3), activation = 'relu', input_shape = c(64, 64, 3)) %>% 
  layer_max_pooling_2d(pool_size=c(2,2)) %>% 
  layer_conv_2d(filters = 32, kernel_size=c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size=c(2,2)) %>% 
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>%
  compile(
    loss = 'binary_crossentropy', 
    optimizer = optimizer_adam(),
    metrics = c('accuracy')
  )
# Data generators and datasets
train_datagen <- image_data_generator(rescale = 1./255,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip = TRUE)
test_datagen <- image_data_generator(rescale = 1./255)
train_dataset <- train_datagen %>% 
  flow_images_from_directory(directory = "dataset/training_set",
                             target_size = c(64, 64),
                             batch_size = 32,
                             class_mode = 'binary' )
test_dataset <- test_datagen %>%
  flow_images_from_directory(directory = 'dataset/test_set',
                             target_size = c(64, 64),
                             batch_size = 32,
                             class_mode = 'binary')
# Fitting the model
model %>% 
  fit_generator(train_dataset, 
                steps_per_epoch = 250, 
                epochs = 10,
                validation_data = test_dataset)