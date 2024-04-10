#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from PIL import Image
import imgaug.augmenters as ia
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

# Define constants
n_classes = 23
input_height = 416
input_width = 608

# Define a learning rate scheduler (example: exponential decay)
learning_rate_initial = 1e-5
decay_rate = 0.96
decay_steps = 1000

# Define paths
train_path = 'semantic_drone_dataset/training_set'
image_folder = 'images'
mask_folder = 'masks'

# Define the Streamlit app
def app():

    original_image = "semantic_drone_dataset/training_set/images/001.jpg"
    label_image_semantic = "semantic_drone_dataset/training_set/masks/001.png"

    fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    axs[0].imshow( Image.open(original_image))
    axs[0].grid(False)

    label_image_semantic = Image.open(label_image_semantic)
    label_image_semantic = np.asarray(label_image_semantic)
    axs[1].imshow(label_image_semantic)
    axs[1].grid(False)
    st.pyplot(fig)


    # Define model and compile
    model = get_unet()

    # Define callbacks
    callbacks = [
        ModelCheckpoint('model_weights.h5', monitor='val_loss', save_best_only=True)
    ]

    # Train the model
    batch_size = 8
    num_epochs = 10

    model.fit(
        generate_data(train_path, image_folder, mask_folder, batch_size),
        steps_per_epoch=len(os.listdir(os.path.join(train_path, image_folder))) // batch_size,
        epochs=num_epochs,
        callbacks=callbacks
    )
    
def learning_rate_schedule(epoch):
    return learning_rate_initial * decay_rate**(epoch // decay_steps)

def get_unet():
    # Define the U-Net model architecture
    learning_rate_scheduler = LearningRateSchedule(learning_rate_schedule)
    # Create the Adam optimizer with the learning rate scheduler
    adam_optimizer = Adam(learning_rate=learning_rate_scheduler)
    inputs = Input((input_height, input_width, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(n_classes, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define a function to generate data
def generate_data(train_path, image_folder, mask_folder, batch_size):
    image_datagen = ImageDataGenerator(rescale=1.0/255)
    mask_datagen = ImageDataGenerator()

    seed = 1

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        target_size=(input_height, input_width),
        batch_size=batch_size,
        seed=seed
    )

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        target_size=(input_height, input_width),
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed
    )

    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        mask = np.expand_dims(mask, axis=-1)
        mask = np.eye(n_classes)[mask]
        yield (img, mask)

# Define a custom callback function to update the Streamlit interface
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get the current loss and accuracy metrics
        loss = logs['loss']
        accuracy = logs['accuracy']
        
        # Update the Streamlit interface with the current epoch's output
        st.write(f"Epoch {epoch}: loss = {loss:.4f}, accuracy = {accuracy:.4f}")

#run the app
if __name__ == "__main__":
    app()
