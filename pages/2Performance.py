#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Define the Streamlit app
def app():

    st.subheader('Testing the Performance of the CNN Classification Model')
    text = """We test our trained model by presenting it with a classification task."""
    st.write(text)
    
    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        present_image(uploaded_file)

def present_image(imagefile):
    model = st.session_state.model

    st.image(imagefile, caption='Uploaded image')
    test_image = image.load_img(imagefile, target_size=(32, 32))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Get the class with the highest probability
    predicted_class = tf.math.argmax(result, axis=1)  # Argmax along axis=1 for class index
    st.write(predicted_class)
    # Get the actual integer index (assuming the first element in result)
    predicted_class_index = int(predicted_class.numpy()[0])
    st.write(predicted_class_index)
    st.write('Predicted class:' + str(class_names[predicted_class_index]))

 
#run the app
if __name__ == "__main__":
    app()
