#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.datasets import cifar10

import time

# Define the Streamlit app
def app():

    if "train_images" not in st.session_state:
        st.session_state.train_images = []
    if "train_labels" not in st.session_state:
        st.session_state.train_labels = []

    text = """Replace with description of CIFAR 10"""
    st.write(text)

    progress_bar = st.progress(0, text="Loading the images, please wait...")

    # Load the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # update the progress bar
    for i in range(100):
        # Update progress bar value
        progress_bar.progress(i + 1)
        # Simulate some time-consuming task (e.g., sleep)
        time.sleep(0.01)
    # Progress bar reaches 100% after the loop completes
    st.success("Image dataset loading completed!") 

    # Create the figure and a grid of subplots
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(6, 8))

    # Iterate through the subplots and plot the images
    for i, ax in enumerate(axes.flat):
        # Turn off ticks and grid
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        # Display the image
        ax.imshow(train_images[500 + i], cmap=plt.cm.binary)
        # Add the image label
        ax.set_xlabel(train_labels[i][0])

    # Show the plot
    plt.tight_layout()  # Adjust spacing between subplots
    st.pyplot(fig)

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    st.session_state.train_images = train_images
    st.session_state.train_labels = train_labels

   # Define CNN parameters    
    st.sidebar.subheader('Set the CNN Parameters')
    options = ["relu", "tanh", "logistic"]
    activation = st.sidebar.selectbox('Select the activation function:', options)

    options = ["adam", "lbfgs", "sgd"]
    optimizer = st.sidebar.selectbox('Select the optimizer:', options)

    hidden_layers = st.sidebar.slider(      
        label="How many hidden layers? :",
        min_value=5,
        max_value=250,
        value=10,  # Initial value
    )

    epochs = st.sidebar.slider(   
        label="Set the epochs:",
        min_value=10,
        max_value=30,
        value=10
    )

    # Convert class labels to one-hot encoded vectors
    num_classes = 10
    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels = keras.utils.to_categorical(test_labels, num_classes)

    # Define the CNN architecture
    model = keras.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    #store the clf object for later use
    st.session_state.model = model

    if st.button('Start Training'):
        progress_bar = st.progress(0, text="Training the MLP regressor can take up to five minutes please wait...")

        # Train the model
        batch_size = 64
        model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, 
                  validation_data=(test_images, test_labels), callbacks=[CustomCallback()])

        #st.session_state.model = model

        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Regressor training completed!") 
        st.write("Use the sidebar to open the Performance page.")

# Define a custom callback function to update the Streamlit interface
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get the current loss and accuracy metrics
        loss = logs['loss']
        accuracy = logs['accuracy']
        
        # Update the Streamlit interface with the current epoch's output
        st.write(f"Epoch {epoch}: loss = {loss:.4f}, accuracy = {accuracy:.4f}")
    
train_images = []
train_labels = []
test_images = []
test_labels = [] 

#run the app
if __name__ == "__main__":
    app()
