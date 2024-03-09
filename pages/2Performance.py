#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import time

# Define the Streamlit app
def app():
    st.subheader('Performance of the DBSCAN Cluster Classifier')
    if st.session_state.new_clusters == True:
        # Create a progress bar object
        progress_bar = st.progress(0, text="Generating performance report, please wait...")
       
        X = st.session_state.X
        y = st.session_state.y
                
        # Define DBSCAN parameters
        eps = 0.3  # Maximum distance between points to be considered neighbors
        min_samples = 10  # Minimum number of neighbors to form a core point

        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_

        # Calculate adjusted Rand score for performance evaluation
        ari = adjusted_rand_score(y, labels)
        print(f"Adjusted Rand Index (ARI): {ari:.3f}")

        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)

        st.session_state.new_clusters = False

        # Create the figure and axes objects
        fig, ax = plt.subplots()

        # Create the scatter plot using ax
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')

        # Add title, labels, and show the plot
        ax.set_title(f"DBSCAN Clustering (ARI: {ari:.3f})")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        st.pyplot(fig)


#run the app
if __name__ == "__main__":
    app()
