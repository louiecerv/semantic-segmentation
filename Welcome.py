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

    text = """The Heirarchical Clustering Algorithm"""
    st.subheader(text)

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('heirarchical-clustering.jpg', caption="Heirarchical Clustering Algorithm""")

    text = """Agglomerative clustering, also known as hierarchical 
    clustering, is a type of clustering that builds a 
    hierarchy of clusters. It follows a bottom-up approach:
    \nStart with individual points: Each data point begins as 
    its own separate cluster.
    \nMerge similar clusters: In each step, the algorithm merges the 
    two most similar clusters based on a distance metric. Repeat until 
    all data points are combined: This continues until all data points 
    are merged into a single cluster. The result is a tree-like 
    structure called a dendrogram that shows how clusters are 
    formed at different levels of similarity. This allows you to 
    decide on a desired level of granularity for your clusters.
    \nBottom-up approach: Starts with individual points and merges 
    them together. Hierarchical structure: Creates a hierarchy of 
    clusters represented by a dendrogram. Doesn't require pre-defined 
    cluster number: You don't need to specify the number of clusters 
    beforehand.
    \nThis clustering method is useful for exploratory 
    data analysis when you want to understand the inherent structure 
    of your data without having to guess the number of clusters upfront."""

    st.write(text)
    
#run the app
if __name__ == "__main__":
    app()
