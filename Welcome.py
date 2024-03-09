#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

# Define the Streamlit app
def app():

    if "new_clusters" not in st.session_state:
        st.session_state.new_clusters = False

    if "n_clusters" not in st.session_state:
        st.session_state.n_clusters = 4

    text = """The DBSCAN, or Density-Based Spatial Clustering of Applications with Noise Clustering Algorithm"""
    st.subheader(text)

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('dbscan.png', caption="DBSCAN Clustering Algorithm""")

    text = """DBSCAN, or Density-Based Spatial Clustering of Applications 
    with Noise, is a popular clustering algorithm used for grouping 
    data points. Unlike K-Means, which requires specifying the 
    number of clusters beforehand, DBSCAN works by identifying 
    dense regions of data points and classifying outliers.
    \nCore points: These are points with a high density of neighbors 
    within a specific radius (epsilon, ε). Imagine a circle around a 
    data point. If the circle contains at least a minimum number of 
    points (minPts), it's a core point.
    \nBorder points: These are points on the fringes of clusters, 
    within the ε distance of a core point, but don't have enough 
    neighbors themselves to be considered core points.
    \nNoise points: These are data points that are far away from any dense 
    region and aren't considered part of any cluster.
    The algorithm iterates through the data points, classifying 
    them as core points, border points, or noise. Here's a simplified 
    view of the process:
    \nPick an unvisited data point.
    \nIf it's a core point, create a new cluster and recursively add 
    its dense neighbors (including border points) to the cluster.
    If it's not a core point (either a border or noise point), 
    mark it as visited and move on.
    \nDBSCAN is effective for finding clusters of various shapes 
    and sizes, especially in datasets with noise and outliers. 
    It's a good choice when the number of clusters is unknown 
    beforehand. However, it requires setting two parameters: ε 
    (radius) and minPts (minimum neighbors), which can impact the 
    clustering results."""

    st.write(text)
    
#run the app
if __name__ == "__main__":
    app()
