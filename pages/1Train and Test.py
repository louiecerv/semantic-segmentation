#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import time
from PIL import Image
import segmentation_models as sm
from segmentation_models import image_augmentation as ia
from tensorflow.keras.optimizers import Adam

# Define the Streamlit app
def app():

    original_image = "semantic_drone_dataset/original_images/001.jpg"
    label_image_semantic = "semantic_drone_dataset/label_images_semantic/001.png"

    fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    axs[0].imshow( Image.open(original_image))
    axs[0].grid(False)

    label_image_semantic = Image.open(label_image_semantic)
    label_image_semantic = np.asarray(label_image_semantic)
    axs[1].imshow(label_image_semantic)
    axs[1].grid(False)
    st.pyplot(fig)

    # Define constants
    epochs = 4
    n_classes = 23
    input_height = 416
    input_width = 608

    # Data augmentation for training images
    train_image_datagen = ia.AugmentationDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        horizontal_flip=True,
        # Add more augmentation parameters as needed
    )

    # No augmentation for masks
    train_mask_datagen = ia.AugmentationDataGenerator(rescale=1.0 / 255)

    # Load training data
    training_set = train_image_datagen.flow_from_directory(
        "semantic_drone_dataset/original_images",
        target_size=(input_height, input_width),
        batch_size=32,
        class_mode=None,
        shuffle=True,  # Shuffle data
    )

    training_mask = train_mask_datagen.flow_from_directory(
        "semantic_drone_dataset/label_images_semantic",
        target_size=(input_height, input_width),
        batch_size=32,
        class_mode=None,
        shuffle=True,  # Shuffle data
    )

    # Define the model
    encoder_name = 'efficientnetb0'
    model = sm.Unet(
        encoder_name=encoder_name,
        encoder_weights='imagenet',
        classes=n_classes,
        activation='softmax'
    )

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    loss = sm.losses.categorical_crossentropy(label_smoothing=0.1)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.categorical_crossentropy]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train the model
    history = model.fit(
        x=training_set,
        y=training_mask,
        batch_size=32,
        epochs=epochs,
        validation_split=0.2
    )

    # Save the best performing model
    model.save('unet_drone_segmentation.h5')




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
