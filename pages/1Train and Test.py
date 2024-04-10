#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from PIL import Image
import imgaug.augmenters as ia
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the Streamlit app
def app():

    original_image = "semantic_drone_dataset/training_set/images/001.jpg"
    label_image_semantic = "semantic_drone_dataset/training_set/masks/001.png"

    fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    axs[0].imshow( Image.open(original_image))
    axs[0].grid(False)

    label_image_semantic = Image.open(label_image_semantic)
    label_image_semantic = np.asarray(label_image_semantic)
    axs[1].imshow(label_image_semantic)
    axs[1].grid(False)
    st.pyplot(fig)

    # Define U-Net model
    def unet(input_height, input_width, n_classes):
        inputs = Input((input_height, input_width, 3))
        
        # Encoder
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        # Bottom
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        drop5 = Dropout(0.5)(conv5)

        # Decoder
        up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(drop5)
        up6 = concatenate([up6, drop4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
        up7 = concatenate([up7, conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
        up8 = concatenate([up8, conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
        up9 = concatenate([up9, conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        # Output layer
        outputs = Conv2D(n_classes, (1, 1), activation='softmax')(conv9)

        model = Model(inputs=[inputs], outputs=[outputs])

        return model

    # Define constants
    epochs = 4
    n_classes = 23
    input_height = 416
    input_width = 608

    # Data augmentation
    train_image_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    training_set = train_image_datagen.flow_from_directory(
        "semantic_drone_dataset/training_set",
        target_size=(64, 64),
        batch_size=32,
        class_mode="input",  # Use "input" for semantic segmentation
        color_mode="rgb",  # Ensure consistent color mode
        classes=['images', 'masks'],  # Assuming your directory structure has images and masks folders
    )

    # Build and compile the model
    model = unet(input_height, input_width, n_classes)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(training_set, steps_per_epoch=len(training_set), epochs=epochs)
  
  




# Define a custom callback function to update the Streamlit interface
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get the current loss and accuracy metrics
        loss = logs['loss']
        accuracy = logs['accuracy']
        
        # Update the Streamlit interface with the current epoch's output
        st.write(f"Epoch {epoch}: loss = {loss:.4f}, accuracy = {accuracy:.4f}")

#run the app
if __name__ == "__main__":
    app()
