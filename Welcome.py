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

    text = """The K-Means Clustering Algorithm"""
    st.subheader(text)

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('k-means.jpeg', caption="K-Means Clustering Algorithm""")

    text = """K-means clustering is a popular unsupervised machine learning algorithm 
    used for clustering data points into groups or clusters based on their similarity. 
    The goal of K-means clustering is to partition the data points into K number of 
    distinct non-overlapping clusters, where K is a pre-defined number specified by the user.
    \nThe algorithm works by first randomly selecting K centroids, which are the initial 
    representative points of the clusters. Then, the algorithm iteratively assigns each 
    data point to the nearest centroid based on the Euclidean distance between the data 
    point and the centroids. After all data points are assigned to a centroid, 
    the algorithm updates the centroid by taking the mean of all data points assigned 
    to it. This process is repeated until convergence, which occurs when the data points 
    no longer change their assignments to centroids.
    \nThe final result of K-means clustering is a set of K clusters, where each data 
    point is assigned to the cluster whose centroid is closest to it. The algorithm is 
    widely used in various fields, including image segmentation, market segmentation, 
    and anomaly detection, among others."""

    st.write(text)
    
#run the app
if __name__ == "__main__":
    app()
