#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import time

# Define the Streamlit app
def app():
    if "X" not in st.session_state: 
        st.session_state.X = []
    
    if "y" not in st.session_state: 
        st.session_state.y = []

    if "scaler" not in st.session_state:
        st.session_state["scaler"] = StandardScaler()

    if "clf" not in st.session_state:
        st.session_state.clf = []

    if "X_train" not in st.session_state:
        st.session_state.X_train = []

    if "X_test" not in st.session_state:
            st.session_state.X_test = []

    if "y_train" not in st.session_state:
            st.session_state.y_train = []

    if "y_test" not in st.session_state:
            st.session_state.y_test = []

    if "X_test_scaled" not in st.session_state:
            st.session_state.X_test_scaled = []

    if "n_clusters" not in st.session_state:
        st.session_state.n_clusters = 4

    text = """Multi-Layer Perceptron Regressor on the California Housing Dataset"""
    st.subheader(text)

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('streetview.png', caption="Street View House Number")

    text = """
    This app utilizes the power of Machine Learning to predict house numbers directly from 
    street view images.
    The aoo will koad the images from the SVHN data in sk-learn.
    Prediction in action: The model built with scikit-learn's MLP Classifier 
    analyzes the image and predicts the house number.
    Machine Learning Model: Multi-Layer Perceptron (MLP) Classifier - a 
    type of artificial neural network trained on a dataset of street view 
    images with labelled house numbers.
    Data Processing: The app pre-processes the uploaded image to ensure 
    compatibility with the model.
    Real-time prediction: The model analyzes the image and outputs 
    the predicted house number.    
    """
    st.text(text)

    text = """Describe the MLP Classifier"""

    st.write(text)


    
#run the app
if __name__ == "__main__":
    app()
