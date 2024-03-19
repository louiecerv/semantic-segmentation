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
from tensorflow.keras.utils import to_categorical


import time

# Define the Streamlit app
def app():

    if "model" not in st.session_state:
        st.session_state.model = []
    
    if "train_images" not in st.session_state:
        st.session_state.training_images = []

    text = """The CIFAR-10 dataset is a collection of 60,000 small, 
    colorful images (32x32 pixels) that belong to 10 distinct categories, 
    like airplanes, cars, and animals. It's a popular choice for 
    training machine learning algorithms, especially those focused on image 
    recognition, because it's easy to access and allows researchers to 
    experiment with different approaches quickly due to the relatively 
    low resolution of the images."""
    st.write(text)

    progress_bar = st.progress(0, text="Loading the images, please wait...")

    # Load the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    #train_images = tf.convert_to_tensor(train_images)
    #train_labels = tf.convert_to_tensor(train_labels)
    #test_images = tf.convert_to_tensor(test_images)
    #test_labels = tf.convert_to_tensor(test_labels)

    #save objects to session state
    st.session_state.training_images = train_images

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    class_indices = dict((v, k) for k, v in enumerate(train_labels[0]))
    st.write(class_indices)

    # Define the class names 
    #class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Print the class name corresponding to the first element in the training set (assuming one-hot encoding)
    #predicted_class = train_labels[0].argmax(axis=0)  # Get index of maximum value
    #st.write('Object classes found in the CIFAR-10 Dataset:')
    #st.write(class_names)

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

    # display images starting with index 500
    start_index = 500
    # Iterate through the subplots and plot the images
    for i, ax in enumerate(axes.flat):
        # Turn off ticks and grid
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        # Display the image
        ax.imshow(train_images[start_index + i], cmap=plt.cm.binary)
        # Add the image label
        ax.set_xlabel(train_labels[i][0])

    # Show the plot
    plt.tight_layout()  # Adjust spacing between subplots
    st.pyplot(fig)

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

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
        min_value=3,
        max_value=30,
        value=3
    )

    # Convert class labels to one-hot encoded vectors
    num_classes = 10
    #train_labels = keras.utils.to_categorical(train_labels, num_classes)
    #test_labels = keras.utils.to_categorical(test_labels, num_classes)

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

    if st.button('Start Training'):
        progress_bar = st.progress(0, text="Training the model please wait...")
        # Train the model
        batch_size = 64
        
        model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, 
                  validation_data=(test_images, test_labels), callbacks=[CustomCallback()])

        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Model training completed!") 
        st.write("Use the sidebar to open the Performance page.")

        # Save the trained model to memory
        st.session_state.model = model

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
