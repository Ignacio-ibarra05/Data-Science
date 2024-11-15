import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

import splitfolders

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, Input, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model

from sklearn.metrics import classification_report, confusion_matrix

# Set parameters
batch = 32
image_size = 64  # Reduced to speed up computation
img_channel = 3
n_classes = 29  # Adjusted to the number of classes

# Initialize ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Define paths
train_path = '/Users/ignacioibarra/Downloads/ASL_Alphabet_Dataset/train'
val_path = '/Users/ignacioibarra/Downloads/ASL_Alphabet_Dataset/val'
test_path = '/Users/ignacioibarra/Downloads/ASL_Alphabet_Dataset/test'

# Create data generators
train_data = datagen.flow_from_directory(
    directory=train_path, 
    target_size=(image_size, image_size), 
    batch_size=batch, 
    class_mode='categorical')

val_data = datagen.flow_from_directory(
    directory=val_path, 
    target_size=(image_size, image_size),
    batch_size=batch, 
    class_mode='categorical')

test_data = datagen.flow_from_directory(
    directory=test_path, 
    target_size=(image_size, image_size), 
    batch_size=batch, 
    class_mode='categorical',
    shuffle=False)

# Define the simplified model
model = Sequential([
    Input(shape=(image_size, image_size, img_channel)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')  # Adjusted to the number of classes
])

# Print the model summary
print(model.summary())

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', 
    min_delta=0.001,
    patience=5,
    restore_best_weights=True, 
    verbose=1)

reduce_learning_rate = ReduceLROnPlateau(
    monitor='val_accuracy', 
    patience=2, 
    factor=0.5, 
    verbose=1)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
asl_class = model.fit(
    train_data, 
    validation_data=val_data,
    epochs=30,
    callbacks=[early_stopping, reduce_learning_rate],
    verbose=1)

# Save the model
model.save('simplified_classification.keras')

# Evaluate for train generator
train_loss, train_acc = model.evaluate(train_data, verbose=0)
print(f'The accuracy of the model for training data is: {train_acc * 100:.2f}%')
print(f'The loss of the model for training data is: {train_loss:.4f}')

# Evaluate for validation generator
val_loss, val_acc = model.evaluate(val_data, verbose=0)
print(f'The accuracy of the model for validation data is: {val_acc * 100:.2f}%')
print(f'The loss of the model for validation data is: {val_loss:.4f}')

# Evaluate for test generator
test_loss, test_acc = model.evaluate(test_data, verbose=0)
print(f'The accuracy of the model for test data is: {test_acc * 100:.2f}%')
print(f'The loss of the model for test data is: {test_loss:.4f}')
