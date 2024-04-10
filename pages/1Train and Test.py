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

    epochs = 4

    import segmentation_models as sm
    from tensorflow.keras.optimizers import Adam  # Or any other optimizer you prefer

    # Define model parameters
    n_classes = 23
    input_height = 416
    input_width = 608

    # Define the model (using 'resnet34' backbone as an example)
    encoder_name = 'resnet34'  # Choose a suitable encoder (e.g., 'vgg16', 'mobilenetv2')
    model = sm.Unet(
        encoder_name=encoder_name,
        encoder_freeze=False,  # Train all layers (adjust based on your needs)
        classes=n_classes,
        activation='softmax'  # Adjust activation based on task (e.g., 'sigmoid' for binary)
    )

    # Define optimizer (replace with your preferred one)
    optimizer = Adam(learning_rate=0.001)

    # Define loss function (replace with your preferred one)
    loss = sm.losses.binary_crossentropy(label_smoothing=0.1)  # Adjust for multi-class

    # Define metrics (replace with your preferred ones)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.categorical_crossentropy]

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Data preparation (modify for your specific data loading)
    # Assuming you have functions to load training images (X_train) and annotations (y_train)
    X_train, y_train = load_training_data(...)

    # Train the model
    model.fit(
        x=X_train,
        y=y_train,
        batch_size=16,  # Adjust batch size based on GPU memory
        epochs=epochs,
        validation_split=0.2  # Split training data for validation
    )

    # Save the trained model (optional)
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
