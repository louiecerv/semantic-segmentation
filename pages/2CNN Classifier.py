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

# Define CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_class(label):
    return class_names[label]

# Define the Streamlit app
def app():

    if "model" not in st.session_state:
        st.session_state.model = []
    
    if "train_images" not in st.session_state:
        st.session_state.training_images = []

    st.header("A Convolutional Neural Network for CIFAR-10 Image Classification")
    text = """This data app trains a Convolutional Neural Network (CNN) on the CIFAR-10 image dataset 
    using TensorFlow. CIFAR-10 consists of 60,000 small colored images from 10 classes, like 
    airplanes and cats. 
    \nThe app will:
    * **Load the CIFAR-10 dataset:** It automatically downloads and prepares the data for training.
    * **Build a CNN model:** You can define the architecture of your CNN, including the number of convolutional layers, filters, and pooling operations.
    * **Train the model:** The app trains the CNN on the training data and monitors its performance. 
    * **Evaluate the model:**  See how well the trained CNN performs on the unseen test data.
    \nThis app is a great tool for:
    * **Learning CNNs:** Experiment with different CNN architectures and understand how they work for image classification.
    * **Getting started with TensorFlow:**  Learn the basics of building and training deep learning models with TensorFlow.
    * **CIFAR-10 image classification research:**  Easily train and evaluate different CNN models on the CIFAR-10 dataset."""
    st.write(text)

    progress_bar = st.progress(0, text="Loading 70,000 images, please wait...")

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Define number of images to display (maximum 25)
    num_images = min(25, train_images.shape[0])

    # Create a grid of subplots
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))  # Adjust figsize for better visualization

    # Reshape images to remove the extra dimension for color channels (assuming RGB)
    train_images_flat = train_images.reshape((train_images.shape[0], -1))

    # Loop through the first num_images and display them on the subplots
    for i in range(num_images):
        # Get the current image and its label
        image = train_images_flat[i].reshape((32, 32, 3))
        label = train_labels[i][0]  # Assuming one-hot encoded labels, extract the class index

        # Display the image and label on the current subplot
        axes[i // 5, i % 5].imshow(image)
        axes[i // 5, i % 5].set_title(get_class(label))
        axes[i // 5, i % 5].axis('off')  # Hide axes for better visualization

    # Show the entire plot
    plt.tight_layout()
    st.pyplot(fig)

    train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalize pixel values

    with st.expander("Click to display the list of classes in the CIFAR-10 Dataset."):
        # Enumerate the classes
        for i, class_name in enumerate(class_names):
            st.write(f"Class {i+1}: {class_name}")

    # update the progress bar
    for i in range(100):
        # Update progress bar value
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    # Progress bar reaches 100% after the loop completes
    st.success("Image dataset loading completed!") 

   # Define CNN parameters    
    st.sidebar.subheader('Set the CNN Parameters')
    options = ["relu", "leaky_relu", "sigmoid"]
    c_activation = st.sidebar.selectbox('Input activation function:', options)

    options = ["softmax", "relu"]
    o_activation = st.sidebar.selectbox('Output activation function:', options)
    
    options = ["adam", "adagrad", "sgd"]
    optimizer = st.sidebar.selectbox('Select the optimizer:', options)

    n_neurons = st.sidebar.slider(      
        label="How many neurons? :",
        min_value=16,
        max_value=128,
        value=128,  # Initial value
        step=16
    )

    epochs = st.sidebar.slider(   
        label="Set the epochs:",
        min_value=5,
        max_value=30,
        value=10
    )

    if st.button('Start Training'):
        progress_bar = st.progress(0, text="Training the model please wait...")
        # Train the model

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation=c_activation, input_shape=(32, 32, 3)),  # First convolutional layer
            layers.MaxPooling2D((2, 2)),  # Downsampling with max pooling
            layers.Conv2D(64, (3, 3), activation=c_activation),  # Second convolutional layer
            layers.MaxPooling2D((2, 2)),  # Further downsampling
            layers.Flatten(),  # Flattening for dense layers
            layers.Dense(n_neurons, activation='relu'),  # Dense layer for classification
            layers.Dense(10, activation=o_activation)  # Output layer with 10 classes
        ])

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(train_images, train_labels, 
            epochs=epochs, batch_size=64, 
            validation_data=(test_images, test_labels), 
            callbacks=[CustomCallback()],)

        # Evaluate the model on the test data
        accuracy = model.evaluate(test_images, test_labels)
        st.write("Test accuracy:", accuracy)

        # Extract loss and accuracy values from history
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        # Create the figure with two side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize for better visualization

        # Plot loss on the first subplot (ax1)
        ax1.plot(train_loss, label='Training Loss')
        ax1.plot(val_loss, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot accuracy on the second subplot (ax2)
        ax2.plot(train_acc, 'g--', label='Training Accuracy')
        ax2.plot(val_acc, 'r--', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Set the main title (optional)
        fig.suptitle('Training and Validation Performance')

        plt.tight_layout()  # Adjust spacing between subplots
        st.pyplot(fig)   

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
