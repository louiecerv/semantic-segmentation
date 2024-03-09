#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import time

# Define the Streamlit app
def app():
    st.subheader('Performance of the K-Means Classifier')
    if st.session_state.new_clusters == True:
        # Create a progress bar object
        progress_bar = st.progress(0, text="Generating performance report, please wait...")
       
        X = st.session_state.X
        y = st.session_state.y
                
        linkage_matrix = linkage(X, 'ward')  # Use Ward linkage for minimizing variance

        # Create a figure and an axes object
        fig, ax = plt.subplots()

        # Plot the dendrogram on the axes object
        dendrogram(linkage_matrix, ax=ax)

        # Set title and labels using the axes object
        ax.set_title("Dendrogram")
        ax.set_xlabel("Data points")
        ax.set_ylabel("Euclidean distance")

        st.pyplot(fig)

        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)

        st.session_state.new_clusters = False

    # Define the number of clusters (k)

    k = st.slider(
        label="Select the number of centroids:",
        min_value=2,
        max_value=10,
        value=4,  # Initial value
    )

    if st.button('Plot'):
        # Create a progress bar object
        progress_bar = st.progress(0, text="Generating random data clusters please wait...")
 
        X = st.session_state.X

        # Perform agglomerative clustering with desired number of clusters
        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = clustering.fit_predict(X)

        # Create the figure and axes
        fig, ax = plt.subplots()

        # Scatter plot the data with labels as color
        ax.scatter(X[:, 0], X[:, 1], c=labels)

        # Add title to the plot
        ax.set_title('Heirarchical Clustering with ' + str(k) + ' clusters')

        st.pyplot(fig)

        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        


#run the app
if __name__ == "__main__":
    app()
