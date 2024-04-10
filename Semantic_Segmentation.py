#Input the relevant libraries
import streamlit as st

# Define the Streamlit app
def app():
    text = """Deep Learning Based Semantic Segmentation"""
    st.subheader(text)

    text = """Prof. Louie F. Cervantes, M. Eng. (Information Engineering) \n
    CCS 229 - Intelligent Systems
    Department of Computer Science
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('semantic-segmentation.jpg', caption="Semantic Segmentation")
    st.write("https://www.tensorflow.org/")
    st.write("https://keras.io/")

    text = """Replace with Description of Semantic Segmentation"""
    st.write(text)
    st.image('cifar10.png', caption="CIFAR-10 Dataset")
    
    with st.expander("How to use this App"):
         text = """Step 1. Go to Training page. Set the parameters of the CNN. Click the button to begin training.
         \nStep 2.  Go to Performance Testing page and click the button to load the image
         and get the model's output on the classification task.
         \nYou can return to the training page to try other combinations of parameters."""
         st.write(text)

    with st.expander("Show more information"):
        text = """
        Replace with details of the process...  
        """
        st.write(text)
    
#run the app
if __name__ == "__main__":
    app()
