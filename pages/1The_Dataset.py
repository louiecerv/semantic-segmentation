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
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

    # Print the first 25 images
    plt.figure(figsize=(6,8))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[500+i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()


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

    # Define the MLP regressor model
    clf = MLPRegressor(solver=solver, 
                        hidden_layer_sizes=(hidden_layers),  
                        activation=activation, 
                        max_iter=max_iter, random_state=42)

    #store the clf object for later use
    st.session_state.clf = clf

    if st.button('Start Training'):
        progress_bar = st.progress(0, text="Training the MLP regressor can take up to five minutes please wait...")

        # Train the model 
        train_model(X_train, y_train)

        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Regressor training completed!") 
        st.write("Use the sidebar to open the Performance page.")

def train_model(X_train_scaled, y_train):
    clf = st.session_state.clf 
    clf.fit(X_train_scaled, y_train)
    st.session_state.clf = clf

#run the app
if __name__ == "__main__":
    app()
