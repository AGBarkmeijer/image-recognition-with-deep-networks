# Title     : MLP MNIST-Classifier (Lab Group 17 - MLHVL)
# Objective : Train a Multi-Layer-Perceptron to classify a set of images,
# as given by the MNIST handwritten numbers dataset.
# Created by: Geanne
# Created on: 11.09.20

# load libraries
devtools::install_github("rstudio/tensorflow")
library(tensorflow)
install_tensorflow(envname = "tf")
library(keras)
library(kerasR)
install_keras()

# load data
mnist <- dataset_mnist()
View(mnist(train))

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Data preperation -model 1
dim(x_train) <- c(nrow(x_train), 784)
dim(x_test) <- c(nrow(x_test), 784)

x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

View(y_train)
View(y_test)


# Model definition -model 1
model <- keras_model_sequential()
model %>%
 layer_dense(units = 256, input_shape = c(784)) %>%
 layer_dense(units = 10, activation = 'softmax')

summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


# Training and evaluation -model 1
history <- model %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 12,
  verbose = 1,
  validation_split = 0.2
)
plot(history)

score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)
score


# Changing model parameters -model 2
model_2 <- keras_model_sequential()
model_2 %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>%
  layer_dense(units = 10, activation = 'softmax')

model_2 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


# Training and evaluation -model 2
history_2 <- model %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 12,
  verbose = 1,
  validation_split = 0.2
)
plot(history_2)

score_2 <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)
score_2


# (NEW TOPIC) Deep convolutional networks
# data preperation -model 3
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

View(y_train)
View(y_test)


# Model definition -model 3
model_3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(28,28,1)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model_3 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)


# Training and evaluation -model 3
history_3 <- model %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 6,
  verbose = 1,
  validation_split = 0.2
)
plot(history_3)

score_3 <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)
score_3


# Add dropout -model 4
model_4 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(28,28,1)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model_4 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)


# Training and evaluation -model 4
history_4 <- model %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 6,
  verbose = 1,
  validation_split = 0.2
)
plot(history_4)

score_4 <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)
score_4

