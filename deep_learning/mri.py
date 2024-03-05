
#dataset taken from kaggle
#dataset link: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data



import warnings
warnings.simplefilter('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pathlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import PIL
from PIL import Image, ImageOps
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras import models

from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


"""

#IMAGE LOADING AND PREPROCESSING STEP


data_path = []


# Set the paths to your dataset folders
train_data_path = "/Users/anuheaparker/Desktop/ml/deep_learning/Training"
test_data_path = "/Users/anuheaparker/Desktop/ml/deep_learning/Testing"

data_path.append(train_data_path)
data_path.append(test_data_path)

images = []
labels = []

# Function to load images and labels
def load_images_and_labels():

    for x in data_path: 


        # Mapping of label names to numeric values
        label_map = {"no_tumor": 0, "meningioma_tumor": 1, "glioma_tumor": 2, "pituitary_tumor": 3}

        # Iterate through each label folder
        for label_folder in os.listdir(x):
            label_folder_path = os.path.join(x, label_folder)
            label = label_map[label_folder]

            # Iterate through each image in the label folder
            for image_file in os.listdir(label_folder_path):
                image_path = os.path.join(label_folder_path, image_file)

                # Read and preprocess the image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                image = cv2.resize(image, (224, 224))  # Resize to desired dimensions

                # Append the image and label to the lists
                images.append(image)
                labels.append(label)

    return np.array(images), np.array(labels)

# Load training images and labels
images, labels = load_images_and_labels()

# Save the preprocessed data (optional)
np.save("images.npy", images)
np.save("labels.npy", labels)

"""


images = np.load("/Users/anuheaparker/Desktop/ml/deep_learning/images.npy")
labels = np.load("/Users/anuheaparker/Desktop/ml/deep_learning/labels.npy")


#plt.imshow(images[0])
#plt.axis("off")
#plt.show()



images, labels = shuffle(images, labels, random_state=101)


#test



def plot_activations_multilayer(num_layers, images_per_row, classifier, activations):
    layer_names = []
    for layer in classifier.layers[:num_layers]:
        layer_names.append(layer.name + ' layer')  # Names of the layers, so you can have them as part of your plot
    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :,
                                                 col * images_per_row + row]
                display_grid[col * size : (col + 1) * size, # Displays the grid
                             row * size : (row + 1) * size] = channel_image
        scale = 2. / size
        plt.figure(figsize=(scale*display_grid.shape[1],
                            scale*display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        



