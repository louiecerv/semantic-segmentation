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

    st.header("Basic Artificial Neural Network for CIFAR-10 Image Classification with TensorFlow")
    text = """This app builds and trains a basic Artificial Neural Network (ANN) model for image 
    classification on the CIFAR-10 dataset using TensorFlow. CIFAR-10 consists of 60,000 
    32x32 pixel color images belonging to 10 classes (e.g., airplanes, cars, cats). The model's 
    performance will be evaluated on a separate test set.
    \n**Model Architecture:**
    \n* The ANN will have a simple structure with:
    \n* **Input Layer:** Flattened image data (32x32x3 pixels = 3072 dimensions).
    \n* **Hidden Layer:** A single dense layer with a user-defined number of neurons and ReLU activation.
    \n* **Output Layer:** 10 neurons with softmax activation for probability distribution across the 10 classes.
    \n**Training Process:**
    \n* The app will:
    \n* Load and pre-process the CIFAR-10 data (normalize pixel values).
    \n* Split the data into training and testing sets.
    \n* Define and compile the ANN model using Adam optimizer and categorical crossentropy loss function.
    \n* Train the model for a specified number of epochs.
    \n* Evaluate the model's accuracy on the test set.
    \n**Performance Measurement:**
    \n* After training, go the performance test page to measure how well the model generalizes to unseen data."""
    st.write(text)
    text = """The CIFAR-10 dataset is a collection of 60,000 small, 
    colorful images (32x32 pixels) that belong to 10 distinct categories, 
    like airplanes, cars, and animals. It's a popular choice for 
    training machine learning algorithms, especially those focused on image 
    recognition, because it's easy to access and allows researchers to 
    experiment with different approaches quickly due to the relatively 
    low resolution of the images."""
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
        #axes[i // 5, i % 5].set_title(f"Class: {label}")
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

    hidden_layers = st.sidebar.slider(      
        label="How many hidden layers? :",
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
            layers.Flatten(input_shape=(32, 32, 3)),  # Flatten input images
            layers.Dense(hidden_layers, activation=c_activation),  # First dense layer with ReLU activation
            layers.Dense(10, activation=o_activation)  # Output layer with 10 classes (CIFAR-10 categories)
        ])

        model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
        )

        history = model.fit(train_images, train_labels, epochs=epochs, batch_size=64, validation_data=(test_images, test_labels), callbacks=[CustomCallback()],)

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
