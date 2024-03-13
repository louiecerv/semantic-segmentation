#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import time

# Define the Streamlit app
def app():
    st.subheader('Performance of the Multi-Layer Perceptron Regressor')

    if st.button('Start'):
        X_test_scaled = st.session_state.X_test_scaled

        # Make predictions on the test set
        y_test_pred = st.session_state.clf.predict(X_test_scaled)
        y_test = st.session_state.y_test

        # Evaluate performance using appropriate metrics (e.g., mean squared error, R-squared)
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)

        st.text("Mean squared error: " + f"{mse:,.2f}")
        st.text("R-squared: " + f"{r2:,.2f}")

        # Create a figure and an axes object
        fig, ax = plt.subplots()

        # Scatter plot using the axes object
        ax.scatter(y_test, y_test_pred, s=5)

        # Set labels and title using the axes object
        ax.set_xlabel("Prices")
        ax.set_ylabel("Predicted Prices")
        ax.set_title("Real vs Predicted Housing Prices")

        # Display the plot
        form2.pyplot(fig)



#run the app
if __name__ == "__main__":
    app()
