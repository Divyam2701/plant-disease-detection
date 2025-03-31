# =============================
# Step 1: Import Required Libraries
# =============================
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# =============================
# Step 2: Load the Machine Learning Model
# =============================
try:
    model = load_model('plant_disease_model.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# =============================
# Step 3: Define Class Names
# =============================
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly_blight', 'Corn-Common_rust')

# =============================
# Step 4: Build the Streamlit App Interface
# =============================
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf to detect disease.")

# Uploading the plant image
plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
submit = st.button('Predict Disease')

# =============================
# Step 5: Handle Prediction Logic
# =============================
if submit:
    if plant_image is not None:
        try:
            # Convert the file to an OpenCV image.
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            if opencv_image is None:
                st.error("Error: Unable to read the image. Please upload a valid image file.")
            else:
                # Display the uploaded image
                st.image(opencv_image, channels="BGR", caption='Uploaded Image')
                st.write(f"Image shape: {opencv_image.shape}")

                # Resize the image to match model's expected input
                opencv_image_resized = cv2.resize(opencv_image, (256, 256))

                # Expand dimensions to match model's input shape (1, 256, 256, 3)
                opencv_image_expanded = np.expand_dims(opencv_image_resized, axis=0)

                # Make Prediction
                Y_pred = model.predict(opencv_image_expanded)

                # Get the class with the highest probability
                predicted_class = CLASS_NAMES[np.argmax(Y_pred)]

                # Safely split the predicted class into components
                class_parts = predicted_class.split('-')
                if len(class_parts) == 2:
                    plant, disease = class_parts
                    st.success(f"This is a **{plant}** leaf with **{disease}**.")
                else:
                    st.warning("Unexpected class format received from the model.")

        except Exception as ex:
            st.error(f"An unexpected error occurred: {ex}")
    else:
        st.warning("Please upload an image before predicting.")
