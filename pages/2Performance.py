#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score
import time

# Define the Streamlit app
def app():
    st.subheader('Performance of the K-Means Classifier')
    if st.session_state.new_clusters == True:
        # Create a progress bar object
        progress_bar = st.progress(0, text="Generating performance report, please wait...")
        n_clusters = st.session_state.n_clusters
        X = st.session_state.X
        y = st.session_state.y
        
        # WCSS (Within Cluster Sum of Squares) list
        wcss_list = []

        # Silhouette score list
        silhouette_scores = []

        # Try different numbers of clusters
        for k in range(2, 11):
            # Create KMeans object
            kmeans = KMeans(n_clusters=k, random_state=0)

            # Fit the model to the data
            kmeans.fit(X)

            # Calculate WCSS
            wcss = kmeans.inertia_
            wcss_list.append(wcss)

            # Calculate Silhouette score
            silhouette_score_val = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(silhouette_score_val)

        # Progress bar reaches 100% after the loop completes
        st.success("Performance data loading completed!")

        # WCSS vs Number of Clusters plot
        fig, ax = plt.subplots()
        ax.plot(range(2, 11), wcss_list)
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("WCSS")
        ax.set_title("WCSS vs Number of Clusters")
        st.pyplot(fig)
        text = """In the k-means algorithm, WCSS (Within Cluster Sum of 
        Squares) is a measure of how well the clusters represent the data 
        within themselves. It refers to the sum of squared distances 
        between each data point and its assigned cluster centroid.
        \nLower WCSS generally indicates better clustering:
        \nTight clusters with small distances between data points and 
        their centroids will contribute less to the WCSS.
        \nConversely, spread-out clusters with larger distances 
        will lead to a higher WCSS."""
        st.write(text)

        # Silhouette Score vs Number of Clusters plot
        fig, ax = plt.subplots()
        ax.plot(range(2, 11), silhouette_scores)
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score vs Number of Clusters")
        st.pyplot(fig)

        text = """The Silhouette Score in k-means clustering is a metric 
        that evaluates how well data points are separated into clusters. 
        It considers both cohesion (similarity within a cluster) 
        and separation (difference between clusters).
        \nRange: -1 to 1
        \nInterpretation:
        \n1: Best case - Data points are tightly packed within 
        their cluster and far from points in other clusters.
        \n0: Indicates indifference - Clusters might overlap or data points 
        aren't clearly separated.
        \n-1: Worst case - Points are assigned to the wrong cluster.
        \nThe Silhouette Score helps you assess the quality of k-means 
        clustering, especially in high-dimensional datasets where 
        visualization isn't feasible."""
        st.write(text)

        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)

        st.session_state.new_clusters = False

    # Define the number of clusters (k)
    k = 4

    k = st.slider(
        label="Select the number of centroids:",
        min_value=2,
        max_value=10,
        value=4,  # Initial value
    )

    if st.button('Plot'):
        # Create a progress bar object
        progress_bar = st.progress(0, text="Generating random data clusters please wait...")
        # Create a KMeans object
        kmeans = KMeans(n_clusters=k)
        # Fit the data to the KMeans model
        X = st.session_state.X
        kmeans.fit(X)

        # Get the cluster labels
        predicted_labels = kmeans.labels_

        # Progress bar reaches 100% after the loop completes
        st.success("Centroids plot completed!")

        fig, ax = plt.subplots()

        # Plot the data with colors corresponding to the predicted labels
        ax.scatter(X[:, 0], X[:, 1], c=predicted_labels)

        # Plot the centroids as points
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red')

        ax.set_title("K-Means Clustering")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        st.pyplot(fig)
        st.write('K-Means Clustering with ' + str(k) + ' clusters')

        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        


#run the app
if __name__ == "__main__":
    app()
