#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs

# Define the Streamlit app
def app():

    st.subheader('The Random CLuster Generator')
    text = """Describe the randon cluster generator"""
    st.write(text)

    n_vlusters = 5
    centers = generate_random_points_in_square(-4, 4, -4, 4, n_clusters)
    X, y = make_blobs(n_samples=n_samples, n_features=2,
                cluster_std=cluster_std, centers = centers,
                random_state=random_state)    

    #plot the generated points
    # Create the figure and axes objects
    fig, ax = plt.subplots()

    # Create the scatter plot using the ax object
    ax.scatter(X[:, 0], X[:, 1], s=50)

    # Customize the plot (optional)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("Scatter Plot Using fig and ax")

    # Show the plot
    st.pyplot(fig)

        

def generate_random_points_in_square(x_min, x_max, y_min, y_max, num_points):
    """
    Generates a NumPy array of random points within a specified square region.

    Args:
        x_min (float): Minimum x-coordinate of the square.
        x_max (float): Maximum x-coordinate of the square.
        y_min (float): Minimum y-coordinate of the square.
        y_max (float): Maximum y-coordinate of the square.
        num_points (int): Number of points to generate.

    Returns:
        numpy.ndarray: A 2D NumPy array of shape (num_points, 2) containing the generated points.
    """

    # Generate random points within the defined square region
    points = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(num_points, 2))

    return points


#run the app
if __name__ == "__main__":
    app()
