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
    st.subheader('Performance of the DBSCAN Cluster Classifier')

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
                
        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_

        # Calculate adjusted Rand score for performance evaluation
        ari = adjusted_rand_score(y, labels)
        st.write(f"Adjusted Rand Index (ARI): {ari:.3f}")
        # Calculate silhouette score
        score = silhouette_score(X, labels)
        st.write(f"Silhouette Score: {score:.3f}")

        # Create the figure and axes objects
        fig, ax = plt.subplots()

        # Create the scatter plot using ax
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')

        # Add title, labels, and show the plot
        ax.set_title(f"DBSCAN Clustering (ARI: {ari:.3f})")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        st.pyplot(fig)
        
        text = """The Adjusted Rand Index (ARI) is a metric used to 
        evaluate the performance of DBSCAN clustering against a ground 
        truth labeling of the data. It measures the agreement between 
        the clustering produced by DBSCAN and the known correct clustering.
        \nHere's how it works in the context of DBSCAN:
        \nAgreement between pairs: ARI considers all pairs of data 
        points. For each pair, it checks if they are assigned to the 
        same cluster in both the DBSCAN results and the ground 
        truth labels.
        \nAdjusted for chance: Unlike the raw Rand Index, ARI accounts 
        for the agreement expected by random chance. This is important 
        because with a large number of data points or clusters, some 
        agreement by chance is inevitable.
        \nScore interpretation: The ARI score ranges from -0.5 to 1. 
        \nA score of 1 indicates perfect agreement between the 
        DBSCAN clustering and the ground truth.
        \nA score close to 0 suggests agreement no better than random clustering.
        \nScores below 0 indicate a particularly bad clustering, where 
        the DBSCAN assignments are significantly worse than random."""
        st.write(text)

        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)

#run the app
if __name__ == "__main__":
    app()
