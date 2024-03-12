#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

# Define the Streamlit app
def app():
    st.subheader('Performance of the Multi-Layer Perceptron Classifier')

    # Define MLP parameters

    options = ["relu", "tanh", "logistic"]
    activation = st.sidebar.selectbox('Select the activation function:', options)

    options = ["lbfgs", "sgd"]
    solver = st.sidebar.selectbox('Select the activation function:', options)

    hidden_layer = st.sidebar.slider(      # Maximum distance between points to be considered neighbors
        label="Select the hidden layer:",
        min_value=5,
        max_value=10,
        value=8,  # Initial value
    )

    alpha = st.sidebar.slider(   # Minimum number of neighbors to form a core point
        label="Set the alpha:",
        min_value=.00001,
        max_value=1.0,
        value=0.001,  # In1.0itial value
    )

    max_iter = st.sidebar.slider(   # Minimum number of neighbors to form a core point
        label="Set the max iterations:",
        min_value=100,
        max_value=10000,
        value=100,  # In1.0itial value
    )

    if st.button('Start'):
        # Create a progress bar object
        progress_bar = st.progress(0, text="Generating performance report, please wait...")
        
        X = st.session_state.X
        y = st.session_state.y

        clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size=0.2, random_state=42)
        
        clf.fit(X_train,y_train)
        y_test_pred = clf.predict(X_test)
        st.subheader('Confusion Matrix')

        st.write('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.text(cm)
        st.subheader('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))
        st.subheader('VIsualization')
        visualize_classifier(clf, X_test, y_test_pred)              

        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)

def visualize_classifier(classifier, X, y, title=''):
    # Define the minimum and maximum values for X and Y
    # that will be used in the mesh grid
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Define the step size to use in plotting the mesh grid 
    mesh_step_size = 0.01

    # Define the mesh grid of X and Y values
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

    # Run the classifier on the mesh grid
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # Reshape the output array
    output = output.reshape(x_vals.shape)
    
    # Create the figure and axes objects
    fig, ax = plt.subplots()

    # Specify the title
    ax.set_title(title)
    
    # Choose a color scheme for the plot
    ax.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
    
    # Overlay the training points on the plot
    ax.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    
    # Specify the boundaries of the plot
    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(y_vals.min(), y_vals.max())
    
    # Specify the ticks on the X and Y axes
    ax.set_xticks(np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0))
    ax.set_yticks(np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0))

    
    st.pyplot(fig)

#run the app
if __name__ == "__main__":
    app()
