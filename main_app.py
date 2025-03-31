import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# =============================
# Step 1: Configure Streamlit Page
# =============================
st.set_page_config(
    page_title="Plant Disease Detection 🌿",
    page_icon="🌱",
    layout="wide"
)

# =============================
# Step 2: Hide Streamlit Branding
# =============================
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        .sidebar .sidebar-content {
            background-color: #262730;
            padding: 10px;
            border-radius: 10px;
        }
        .sidebar button {
            width: 100%;
            text-align: left;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            border: none;
            background: #4CAF50;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }
        .sidebar button:hover {
            background: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# =============================
# Step 3: Login Page (Without Backend)
# =============================
def login_page():
    st.title("🔒 Login to Plant Disease Detection")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    
    if st.button("Login"):
        if username == "admin" and password == "password":
            st.session_state["logged_in"] = True
        else:
            st.error("❌ Incorrect username or password")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
    st.stop()

# =============================
# Step 4: Sidebar for Navigation
# =============================
st.sidebar.title("🌱 Plant Detection Menu")
st.sidebar.button("🏠 Home")
st.sidebar.button("🔍 Detect Disease")
st.sidebar.button("📜 About")
st.sidebar.button("🔓 Logout", on_click=lambda: st.session_state.update({"logged_in": False}))

# =============================
# Step 5: Load the Machine Learning Model
# =============================
@st.cache_resource
def load_trained_model():
    try:
        return load_model('plant_disease_model.h5')
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_trained_model()

# =============================
# Step 6: Define Class Names
# =============================
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Early_blight', 'Corn-Common_rust')

# =============================
# Step 7: Build the Streamlit App Interface
# =============================
st.title("🌿 Plant Disease Detection")
st.markdown("Upload an image of the plant leaf to detect possible diseases.")

# Uploading the plant image
plant_image = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])
submit = st.button('🔍 Predict Disease')

# =============================
# Step 8: Handle Prediction Logic
# =============================
if submit:
    if plant_image is not None:
        try:
            # Convert the file to an OpenCV image
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            if opencv_image is None:
                st.error("❌ Error: Unable to read the image. Please upload a valid image file.")
            else:
                # Display the uploaded image
                st.image(opencv_image, channels="BGR", caption="📷 Uploaded Image", use_container_width=True)
                
                # Resize the image to model's expected input
                opencv_image_resized = cv2.resize(opencv_image, (256, 256))

                # Expand dimensions to match model's input shape (1, 256, 256, 3)
                opencv_image_expanded = np.expand_dims(opencv_image_resized, axis=0)

                # Show loading spinner while predicting
                with st.spinner("🔄 Detecting disease... Please wait."):
                    Y_pred = model.predict(opencv_image_expanded)
                
                # Get the class with the highest probability
                predicted_class = CLASS_NAMES[np.argmax(Y_pred)]
                
                # Extract plant and disease name
                class_parts = predicted_class.split('-')
                if len(class_parts) == 2:
                    plant, disease = class_parts
                    st.success(f"✅ This is a **{plant}** leaf with **{disease}**.")
                else:
                    st.warning("⚠️ Unexpected class format received from the model.")

        except Exception as ex:
            st.error(f"🚨 An unexpected error occurred: {ex}")
    else:
        st.warning("⚠️ Please upload an image before predicting.")

# =============================
# Step 9: About Section
# =============================
st.markdown("""
## 📜 About Plant Disease Detection
This application uses deep learning to detect plant diseases from leaf images. 

### Features:
- Upload a plant leaf image.
- AI-powered model predicts possible diseases.
- Supports Tomato, Potato, and Corn leaf diseases.
- Easy-to-use interface.

🔬 **How it works?**
1. The image is preprocessed and analyzed using a CNN model.
2. The model predicts the type of disease.
3. Results are displayed with high accuracy.

This tool helps farmers and researchers detect plant diseases early and take necessary actions to prevent crop damage.
""", unsafe_allow_html=True)
