
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
from sklearn.metrics import confusion_matrix

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

from collections import Counter


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



#helper function to see the activation layers

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
        plt.imshow(display_grid, aspect='auto', cmap='flare')
        plt.show()
















#load in the processed images 
images = np.load("/Users/anuheaparker/Desktop/ml/deep_learning/images.npy")
labels = np.load("/Users/anuheaparker/Desktop/ml/deep_learning/labels.npy")


#plt.imshow(images[0])
#plt.axis("off")
#plt.show()

#look at the number of images per tumor category 
label_counts = Counter(labels)
for category, count in label_counts.items():
    print(f"Category {category}: {count} images")

#split the dataset into train and test data 
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=101)

"""

#MODEl #1

model_1 = Sequential()

model_1.add(Conv2D(32, (5,5), padding='same', input_shape=(224,224,3), activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model_1.add(Conv2D(32, (3, 3), padding='same', activation = 'relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model_1.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model_1.add(Flatten())
model_1.add(Dense(units=512, activation='relu'))
model_1.add(Dense(units=5, activation='softmax'))

model_1.build((1,224, 224,3))
model_1.summary()

img_tensor = np.array(images_train[20], dtype='int')
plt.imshow(img_tensor)
plt.show()

img_tensor = np.expand_dims(img_tensor, axis=0)
y = model_1.predict(img_tensor)
print(f"The predicted output of the sample image has a shape of {y.shape}.")

layer_outputs = [layer.output for layer in model_1.layers] 
activation_model = Model(inputs=model_1.input, outputs=layer_outputs) 
activations = activation_model.predict(img_tensor)

plot_activations_multilayer(8, 8, model_1, activations)

model_1.compile(optimizer='adam', 
            loss="sparse_categorical_crossentropy", 
            metrics=['accuracy'])

model_1.fit(
    images_train, 
    labels_train, 
    epochs=10, 
    batch_size=64
)

layer_outputs = [layer.output for layer in model_1.layers]
activation_model = Model(inputs=model_1.input, outputs=layer_outputs)

loss_1, accuracy_1 = model_1.evaluate(images_test, labels_test)
print(f'Model 1 Test loss: {loss_1}, Model 1 Test accuracy: {accuracy_1}')


"""



"""

#MODEl #2

model_2 = Sequential()

model_2.add(Conv2D(16, (3, 3), padding='same', input_shape=(224, 224, 3), activation='relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2)))
model_2.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_2.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_2.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_2.add(Flatten())
model_2.add(Dense(units=256, activation='relu'))
model_2.add(Dense(units=5, activation='softmax'))

model_2.build((1,224, 224,3))
model_2.summary()

img_tensor = np.array(images_train[20], dtype='int')
plt.imshow(img_tensor)
plt.show()

img_tensor = np.expand_dims(img_tensor, axis=0)
y = model_2.predict(img_tensor)
print(f"The predicted output of the sample image has a shape of {y.shape}.")

layer_outputs = [layer.output for layer in model_2.layers] 
activation_model = Model(inputs=model_2.input, outputs=layer_outputs) 
activations = activation_model.predict(img_tensor)

plot_activations_multilayer(8, 8, model_2, activations)

model_2.compile(optimizer='adam', 
            loss="sparse_categorical_crossentropy", 
            metrics=['accuracy'])

model_2.fit(
    images_train, 
    labels_train, 
    epochs=10, 
    batch_size=64
)

layer_outputs = [layer.output for layer in model_2.layers]
activation_model = Model(inputs=model_2.input, outputs=layer_outputs)

loss_2, accuracy_2 = model_2.evaluate(images_test, labels_test)
print(f'Model 2 Test loss: {loss_2}, Model 2 Test accuracy: {accuracy_2}')

"""







#MODEl #3

model_3 = Sequential()

model_3.add(Conv2D(16, (3, 3), padding='same', input_shape=(224, 224, 3), activation='relu'))
model_3.add(MaxPooling2D(pool_size=(2, 2)))
model_3.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model_3.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_3.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
model_3.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_3.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
model_3.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_3.add(Flatten())
model_3.add(Dense(units=128, activation='relu'))
model_3.add(Dense(units=5, activation='softmax'))

model_3.build((1,224, 224,3))
model_3.summary()

img_tensor = np.array(images_train[20], dtype='int')
plt.imshow(img_tensor)
plt.show()

img_tensor = np.expand_dims(img_tensor, axis=0)
y = model_3.predict(img_tensor)
print(f"The predicted output of the sample image has a shape of {y.shape}.")

layer_outputs = [layer.output for layer in model_3.layers] 
activation_model = Model(inputs=model_3.input, outputs=layer_outputs) 
activations = activation_model.predict(img_tensor)

plot_activations_multilayer(8, 8, model_3, activations)

model_3.compile(optimizer='adam', 
            loss="sparse_categorical_crossentropy", 
            metrics=['accuracy'])

model_3.fit(
    images_train, 
    labels_train, 
    epochs=10, 
    batch_size=64
)

layer_outputs = [layer.output for layer in model_3.layers]
activation_model = Model(inputs=model_3.input, outputs=layer_outputs)

loss_3, accuracy_3 = model_3.evaluate(images_test, labels_test)
print(f'Model 3 Test loss: {loss_3}, Model 3 Test accuracy: {accuracy_3}')







"""

#CODE FOR A CONFUSION MATRIX - only looking at Model 1 

# Make predictions on test data
predictions = model_1.predict(images_test)

# Convert probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(labels_test, predicted_labels)

# Calculate sensitivity and specificity
true_positives = np.diag(conf_matrix)
false_negatives = np.sum(conf_matrix, axis=1) - true_positives
false_positives = np.sum(conf_matrix, axis=0) - true_positives
true_negatives = np.sum(conf_matrix) - (true_positives + false_negatives + false_positives)

sensitivity = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positives)

print("Sensitivity:", sensitivity)
print("Specificity:", specificity)


classes = ["No Tumor", "Meningioma", "Glioma", "Pituitary"]

plt.figure(figsize=(8,6))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='flare', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()



"""







"""

#LIME 
#sample code, not fully written yet 

import lime
import lime.lime_image
import numpy as np
import matplotlib.pyplot as plt

# Initialize LIME explainer
explainer = lime.lime_image.LimeImageExplainer()

# Choose a sample image for explanation
sample_index = 0
sample_image = images_test[sample_index]

# Define the CNN model prediction function
def cnn_predict(image):
    # Normalize image to range [0, 1]
    normalized_image = image / 255.0
    # Preprocess image (e.g., resizing if needed)
    preprocessed_image = preprocess_image(normalized_image)
    # Make predictions using the CNN model
    predictions = model_1.predict(np.array([preprocessed_image]))
    return predictions.flatten()

# Generate explanations using LIME
explanation = explainer.explain_instance(sample_image, cnn_predict, top_labels=1, hide_color=0, num_samples=1000)

# Show explanation
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()


"""