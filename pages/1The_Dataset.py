#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


import time

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


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

    progress_bar = st.progress(0, text="Loading 70,000 images, please wait...")

    (ds_train, ds_test), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train1 = ds_train.batch(25)  # Batch for efficient loading
    # Get a batch of 25 random images
    images, _ = next(iter(ds_train1))

    # Convert TensorFlow images to NumPy arrays
    images_np = images.numpy()  # Assuming images are in the format (batch_size, height, width, channels)

    # Reshape NumPy arrays for display
    images_reshaped = images_np.reshape((25, *images_np.shape[1:4]))  # Reshape to (25, height, width, channels)

    # Display the images using st.image
    col1, col2, col3, col4, col5 = st.columns(5)  # Create 5 columns for layout

    # Loop through each image and display in a column
    for i in range(5):
        for j in range(5):
            image_index = i * 5 + j
            if image_index < 25:
                col = locals()[f'col{j+1}']  # Dynamically access columns
                col.image(images_reshaped[image_index], width=100)  # Adjust width as needed

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


    with st.expander("Click to display the list of classes in the CIFAR-10 Dataset."):
        # Define CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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

    hidden_layers = st.sidebar.slider(      
        label="How many hidden layers? :",
        min_value=16,
        max_value=128,
        value=64,  # Initial value
        step=16
    )

    epochs = st.sidebar.slider(   
        label="Set the epochs:",
        min_value=3,
        max_value=30,
        value=3
    )


    if st.button('Start Training'):
        progress_bar = st.progress(0, text="Training the model please wait...")
        # Train the model
        batch_size = 64

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation=c_activation, input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=c_activation),
            tf.keras.layers.Dense(10, activation=o_activation)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), 'accuracy']
        )

        history = model.fit(
            ds_train,
            epochs=epochs,
            validation_data=ds_test,
            callbacks=[CustomCallback()],
        )

        # Evaluate the model on the test data
        SparseCategoricalAccuracy = model.evaluate(ds_test)
        st.write("Test accuracy:", SparseCategoricalAccuracy)

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
