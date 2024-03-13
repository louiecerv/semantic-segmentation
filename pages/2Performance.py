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
    text = """Test the performance of the MLP Regressor using the 20% of the dataset that was
    set aside for testing. Mean squarer Errir (MSE) and the R-sqaured are the metrics,"
    
    if st.button('Begin Test'):
        
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
        st.pyplot(fig)

        text = """An R-squared of 0.71 on the California housing dataset indicates that 71% of the variance 
        in the median house prices can be explained by the features included in the model. In other words, 
        the model captures a significant portion of the factors influencing housing prices in California.
        The model can be used to predict median house prices based on the features it considers. However, 
        it's important to understand the limitations of relying solely on R-squared:
        Higher R-squared doesn't guarantee perfect predictions: A value of 0.71 signifies a good fit, 
        but there's still 29% of the variance unexplained by the model. This means the model's predictions 
        will not be completely accurate and will have some degree of error."""

        st.write(text)



#run the app
if __name__ == "__main__":
    app()
