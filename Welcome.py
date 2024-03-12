#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

import time

# Define the Streamlit app
def app():

    if "n_clusters" not in st.session_state:
        st.session_state.n_clusters = 4

    text = """Multi-Layer Perceptron Artificial Neural Network Classifier"""
    st.subheader(text)

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('mlp.png', caption="Multilayer Perceptron Neural Network")

    text = """SKlearn's MLP classifier, also called  Multi-layer Perceptron 
    Classifier (MLPClassifier), is a tool for building artificial neural networks 
    for classification tasks. 
    \nCore functionality: Implements a Multi-layer Perceptron (MLP) algorithm, 
    a type of artificial neural network. Learns a function that maps input data 
    to a specific category during training on a dataset.
    \nKey characteristics:
    Uses backpropagation for training, an iterative process to adjust the 
    network's internal parameters.  Supports only the cross-entropy loss function, 
    suitable for tasks with multiple class outputs. 
    Applies softmax activation for the output layer, enabling probability 
    estimations for each class.
    \nAdvantages:
    Can learn complex relationships between features and target variables, 
    making it suitable for non-linear problems.
    \nComputationally expensive to train compared to simpler algorithms.
    \nRequires careful hyperparameter tuning to achieve optimal performance."""

    st.write(text)
    
#run the app
if __name__ == "__main__":
    app()
