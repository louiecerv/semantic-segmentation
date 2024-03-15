#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import time

# Define the Streamlit app
def app():

    text = """Replace with description of CIFAR 10"""
    st.write(text)

    # Load the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


    # Define class names
    #class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck']

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
        #ax.set_xlabel(feature_names[train_labels[i][0]])
        ax.set_xlabel(train_labels[i][0])

    # Show the plot
    plt.tight_layout()  # Adjust spacing between subplots
    st.pyplot(fig)


    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

   # Define MLP parameters    
    st.sidebar.subheader('Set the MLP Parameters')
    options = ["relu", "tanh", "logistic"]
    activation = st.sidebar.selectbox('Select the activation function:', options)

    options = ["adam", "lbfgs", "sgd"]
    solver = st.sidebar.selectbox('Select the solver:', options)

    hidden_layers = st.sidebar.slider(      
        label="How many hidden layers? :",
        min_value=5,
        max_value=250,
        value=10,  # Initial value
    )

    alpha = st.sidebar.slider(   
        label="Set the alpha:",
        min_value=.001,
        max_value=1.0,
        value=0.1,  # In1.0itial value
    )

    max_iter = st.sidebar.slider(   
        label="Set the max iterations:",
        min_value=100,
        max_value=300,
        value=120,  
        step=10
    )

    # Define the CNN architecture
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    # Compile the model
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    #store the clf object for later use
    st.session_state.model = model



    if st.button('Start Training'):
        progress_bar = st.progress(0, text="Training the MLP regressor can take up to five minutes please wait...")

        train_model(model)

        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Regressor training completed!") 
        st.write("Use the sidebar to open the Performance page.")

def train_model(model):
    # Train the model
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))

#run the app
if __name__ == "__main__":
    app()
