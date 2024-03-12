#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
import time

# Define the Streamlit app
def app():

    if "X" not in st.session_state: 
        st.session_state.X = []
    
    if "y" not in st.session_state: 
        st.session_state.y = []

    st.subheader('The Random Cluster Generator')
    text = """Generating data points with clusters:
    \nUses make_blobs function from scikit-learn to 
    generate sample data with desired characteristics:
    n_samples: number of data points.
    n_features: 2 features, representing a 2-dimensional dataset.
    cluster_std: standard deviation for cluster dispersion.
    centers: The generated random centers for each cluster.
    random_state: 42 for reproducibility.
    \nOutput:
    X: A 2-dimensional array containing the generated data points, 
    where each row represents a data point with 2 features.
    y: An array containing the assigned cluster labels for 
    each data point, corresponding to the n_clusters clusters."""
    st.write(text)

    n_clusters = st.sidebar.slider(
    label="Number of clusters:",
    min_value= 2,
    max_value= 10,
    value=4,  # Initial value
    )

    n_samples = st.sidebar.slider(
    label="Select the number of samples:",
    min_value=10,
    max_value=4000,
    value=500,  # Initial value
    )

    cluster_std = st.sidebar.slider(
    label="Select the cluster std:",
    min_value= 0.2,
    max_value= 2.0,
    value=0.5,  # Initial value
    )

    if st.button("Generate"):

        # Create a progress bar object
        progress_bar = st.progress(0, text="Generating random data clusters please wait...")

        st.session_state.n_clusters = n_clusters
        random_state = 42
        centers = generate_random_points_in_square(-4, 4, -4, 4, n_clusters)
        X, y = make_blobs(n_samples=n_samples, n_features=2,
                    cluster_std=cluster_std, centers = centers,
                    random_state=random_state)    

        st.session_state.X = X
        st.session_state.y = y

        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)

        # Progress bar reaches 100% after the loop completes
        st.success("Data clusters loading completed!")

        #plot the generated points
        # Create the figure and axes objects
        fig, ax = plt.subplots()

        # Create the scatter plot using the ax object
        ax.scatter(X[:, 0], X[:, 1], s=50)

        # Customize the plot (optional)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_title("Randomly generated cluster of points")

        # Show the plot
        st.pyplot(fig)

        st.write('Click the Generate button to generate new data clusters.')
        st.write('Navigate to the Performance Page in the sidebar to view the perforance report.')

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
