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
