# Title     : MLP MNIST-Classifier (Lab Group 17 - MLHVL)
# Objective : Train a Multi-Layer-Perceptron to classify a set of images,
# as given by the MNIST handwritten numbers dataset.
# Created by: mik
# Created on: 04.09.20

# load libraries
library(keras)
library(kerasR)
install_keras()

# load data
mnist <- dataset_mnist()

# Part: Model definition
# create model
model <- keras_model_sequential()
model %>%
 layer_dense(units = 256, input_shape = c(784)) %>%
 layer_dense(units = 10, activation = 'softmax')