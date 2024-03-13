#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
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


    # Load the California housing data
    data = fetch_california_housing()

    # Convert data features to a DataFrame
    feature_names = data.feature_names
    st.write(feature_names)
    df = pd.DataFrame(data.data, columns=feature_names)
    df['target'] = data.target
    
    st.write('The California Housing Dataset')
    st.write(df)

    # Separate features and target variable
    X = df.drop('target', axis=1)  # Target variable column name
    y = df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # store for later use
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    # Standardize features using StandardScaler (recommended)
    scaler = st.session_state["scaler"] 
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # save the scaler object for later use
    st.session_state["scaler"] = scaler

    # Define the MLP regressor model
    clf = MLPRegressor(solver='lbfgs',  # Choose a suitable solver (e.g., 'adam')
                        hidden_layer_sizes=(100, 50),  # Adjust hidden layer sizes
                        activation='relu',  # Choose an activation function
                        random_state=42)

    
    progress_bar = st.progress(0, text="Training the MLP regressor please wait...")
    # Train the model
    #clf.fit(X_train_scaled, y_train)
    # update the progress bar
    for i in range(100):
        # Update progress bar value
        progress_bar.progress(i + 1)
        # Simulate some time-consuming task (e.g., sleep)
        time.sleep(0.01)
    # Progress bar reaches 100% after the loop completes
    st.success("Regressor training completed!") 

    #store the clf object for later use
    st.session_state.clf = clf

    plot_feature(df["MedInc"], df["target"], 
                 "Median Income (Thousands USD)", 
                 "Median House Value (Thousands USD", 
                 "Median Income vs. Median House Value")

    plot_feature(df["HouseAge"], df["target"], 
                 "House Age in years", 
                 "Median House Value (Thousands USD", 
                 "House Age vs. Median House Value")
    
    plot_feature(df["AveRooms"], df["target"], 
                 "Average Rooms", 
                 "Median House Value (Thousands USD", 
                 "Average vs. Median House Value")
        
    plot_feature(df["AveBedrms"], df["target"], 
                 "Average Bed Rooms", 
                 "Median House Value (Thousands USD", 
                 "Average Bed Rooms vs. Median House Value")
    
    plot_feature(df["Population"], df["target"], 
                 "Population", 
                 "Median House Value (Thousands USD", 
                 "Population vs. Median House Value")

def plot_feature(feature, target, labelx, labely, title):
    # Display the plots
    fig, ax = plt.subplots(figsize=(10, 6))
    # Scatter plot
    ax.scatter(feature, target)
    # Add labels and title
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    ax.set_title(title)
    # Add grid
    ax.grid(True)
    st.pyplot(fig)

#run the app
if __name__ == "__main__":
    app()
