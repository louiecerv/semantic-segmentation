#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import time

# Define the Streamlit app
def app():

    text = """The California housing dataset is a popular benchmark dataset frequently 
    used in the field of regression analysis for real estate price prediction. 
    It contains information about various features that potentially influence 
    housing prices in California.
    Data Description:
    \nSource: Publicly available from https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
    Size: Contains 20,640 data points, each representing a block in California."""
    st.write(text)

    # Load CIFAR-10 data
    cifar = fetch_openml('cifar_10', version=1)
    X = cifar.data.astype('float32') / 255  # Normalize pixel values

    # One-hot encode target labels
    y = cifar.target.astype('int')
    y = np.eye(10)[y]  # One-hot encoding

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # store for later use
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

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
