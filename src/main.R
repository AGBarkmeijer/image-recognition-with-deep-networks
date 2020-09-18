# Title     : MLP MNIST-Classifier (Lab Group 17 - MLHVL)
# Objective : Train a Multi-Layer-Perceptron to classify a set of images,
# as given by the MNIST handwritten numbers dataset.
# Created by: mik
# Created on: 04.09.20

# load libraries
library(keras)
library(kerasR)
install_keras()

act_fun_mlp = "linear"

# load data
mnist <- dataset_mnist()

# reshape data
x_train <- array_reshape(mnist$train$x, c(60000, 28*28))
x_test <- array_reshape(mnist$test$x, c(10000, 28*28))

x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(mnist$train$y)
y_test <- to_categorical(mnist$test$y)

# Part: Model definition
# create model
model <- keras_model_sequential()
model %>%
 layer_dense(units = 256, input_shape = c(784), activation = act_fun_mlp) %>%
 layer_dense(units = 10, activation = 'softmax')

# compile model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# training
history <- model %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 12,
  verbose = 1,
  validation_split = 0.2
)

# Question 2
# All epochs 2s, epoch 1 7ms/step, epoch 2-12 5ms/step
# Training performance: 
# Accuracy: Start at 0.89, increase to 0.91 after second epoch, afterwards relatively steady at 0.92
# Validation accuracy: Steady at 0.92
# Loss: STarting at 0.4, steep decrease to 0.31 after second epoch, then steadily decreasing to 0,27
# Validation Loss: Relatively steady at 0.28 with spikes to 0.29 at epoch 5 and 11

# Question 3 
# Look at graph in viewer

# Question 4:
# Huge progress until epoch 3 
# -> Model generalizes well until then afterwards not much generalizaion happening
# Accuracy on validation set already high in the very beginning
# -> Model does not generalize at all as there is no performance increase in validation set?
# -> NO: Validation set is not test set
# Just by looking at training set we cannot say whether the model generalizes
# -> We have to look at test (real world) set

# evaluation
score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)

# Question 5:
score

# Question 6:
# Accuracy = (TP + TN) / ALL
# Accuracy might be sufficient as long as there is no trade off between different kinds of errors
# If you want to automatically digitize handwritten criminal records then the weight for false positives (wrong letter identified) very huge as there is nobody who checks the document manually and detects the error.
# Actually precision would give the same values as accuracy as classification is one hot encoded
# -> When an instance is true positive it is also true negative for every other class
# Of course an accuracy of 0.88 is not sufficient for most tasks

# Question 7:
# With a linear activation function there is no non linearity introduced into the model
# -> Not linear relationships between input an output cannot be detected

# Question 8
# show viewer

# Question 9:
# Similar training times for each epoch
# Much higher increase in performance after epoch 2
# Accuracy almost at 1
# Loss much smaller
# Difference between validation and test set is bigger
# -> Model generalizes much better
# -> Not saturated after epoch 2

x_train <- array_reshape(mnist$train$x, c(60000, 28, 28, 1)) / 255
x_test <- array_reshape(mnist$test$x, c(10000, 28, 28, 1)) / 255

y_train <- to_categorical(mnist$train$y)
y_test <- to_categorical(mnist$test$y)

# model definition
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu', input_shape = c(28,28,1)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

# compile model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# fit model
model %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 6,
  verbose = 1,
  validation_split = 0.2
)

# Question 10:
# Show view

# Question 11:
# Much steeper increase in performance from epoch 1 to epoch 2
# In general higher values in accuracy as well as lower values in loss
# Smaller difference between validation and train data

# evaluation
score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)

# Question 12:
# Lesser difference between training score and test score
# -> Higher generalization / performance in deployment
score

# Question 13:
# Even 0.98 is not sufficient for some tasks where the detection must be 100% reliable to prevent conversion errors
# -> One wrong letter in digitization changes the words meaning a lot
# -> For other applications this is acceptable

# Question 14:
# Worse performance on training set but same generalization on test data
# Training time increased from ~ 60 sec to ~ 72 sec

# Question 15:
# It seems like the model without dropout generalizes better because metrics are higher but the model with dropout will perform better in a real world environment where data might look different
# -> It is more robust and generalizes better as well
# -> But Accuracy metric does not measure generalization but rather performance on a predefined set

cifar10 <- dataset_cifar10()

x_train <- cifar10$train$x / 255
x_test <- cifar10$test$x / 255

y_train <- to_categorical(cifar10$train$y)
y_test <- to_categorical(cifar10$test$y)

##################################### model task 2 ########################################
model_task-2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu', input_shape = c(32,32,3), padding="same") %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu', padding="same") %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

# compile model
model_task_2 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = c('accuracy')
)

# fit model
model_task_2 %>% fit(
  x_train, y_train,
  batch_size = 32,
  epochs = 20,
  verbose = 1,
  validation_data = list(x_test, y_test),
  shuffle = TRUE
)
