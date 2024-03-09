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

# Define the Streamlit app
def app():
    st.subheader('Performance of the K-Means Classifier')
    n_clusters = st.session_state.n_clusters
    X = st.session_state.X
    y = st.session_state.y
    
    clf = KMeans(n_clusters=n_clusters)
    clf.fit(X)
    y_test_pred = clf.predict(X)

    centers, labels = find_clusters(X, n_clusters)

def find_clusters(X, n_clusters, rseed=42):
    #randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    # Create the plot
    fig, ax = plt.subplots()

    # Scatter plot for data points
    ax.scatter(X[:, 0], X[:, 1], c=y_test_pred, s=50, cmap='bright')

    # Scatter plot for centers
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)

    # Display the plot using Streamlit
    st.pyplot(fig)

    while True:
        labels = pairwise_distances_argmin(X, centers)
        new_centers = np.array([X[labels==i].mean(0) for i in range(n_clusters)])

        #check for convergence
        if np.all(centers==new_centers):
            break
        centers = new_centers
    return centers, labels

#run the app
if __name__ == "__main__":
    app()
