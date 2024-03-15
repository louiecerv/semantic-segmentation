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

    if "model" not in st.session_state:
        st.session_state.model = []

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

    text = """Convolutional Neural Network on the CIFAR-10 Dataset"""
    st.subheader(text)

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('cifar10.png', caption="CIFAR-10 Dataset")

    text = """
    This Streamlit application utilizes a pre-existing dataset called CIFAR-10 
    for image classification using a Convolutional Neural Network (CNN).
    Data Source:
    The application relies on the CIFAR-10 dataset, a well-known benchmark 
    dataset for image classification tasks.
    This dataset is publicly available and can be downloaded directly within 
    the application code using TensorFlow's datasets.cifar10.load_data() function.
    Data Description:
    CIFAR-10 consists of 60,000 color images in 32x32 pixel resolution.
    
    The images are categorized into 10 mutually exclusive classes:
    airplane
    automobile
    bird
    cat
    deer
    dog
    frog
    horse
    ship
    truck
    The data is split into 50,000 training images and 10,000 testing images.
    This split allows the CNN model to learn from the training data and evaluate 
    its performance on unseen data (testing data).
    
    Data Preprocessing (performed within the application):
    The application performs minimal preprocessing on the data:
    Normalization: Pixel values are typically normalized to a range between 0 and 1 
    for better training efficiency.
    Data Used in the App:
    
    The application utilizes the following data from CIFAR-10:
    Training images: Used to train the CNN model to recognize patterns and features.
    Training labels: These labels correspond to the class categories of each training image.
    Test images: Used to evaluate the trained model's performance on unseen data.
    Test labels: Labels corresponding to the classes of the test images.    
    """
    st.write(text)
    
#run the app
if __name__ == "__main__":
    app()
