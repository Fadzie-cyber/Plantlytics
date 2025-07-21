import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('model3.h5')

# Define your class labels
CLASS_NAMES = [
    'Cashew_Anthracnose','Cashew_Gumosis','Cashew_Healthy','Cashew_Leafminer','Cashew_Red Rust',
    'Cashew_anthracnose','Cashew_gumosis','Cashew_healthy','Cashew_leafminer','Cashew_red rust',
    'Cassava_Bacterial Blight','Cassava_Bacterial Blight3241','Cassava_Brown Spot','Cassava_Green Mite','Cassava_Healthy','Cassava_Mosaic',
    'Cassava_bacterial blight','Cassava_bacterial blight3241','Cassava_brown spot','Cassava_green mite','Cassava_healthy','Cassava_mosaic',
    'Maize_Fall Armyworm','Maize_Grasshopper','Maize_Healthy','Maize_Leaf Beatle','Maize_Leaf Blight','Maize_Leaf Spot','Maize_Streak Virus',
    'Maize_fall armyworm','Maize_grasshopper','Maize_healthy','Maize_leaf beatle','Maize_leaf blight','Maize_leaf spot','Maize_streak virus',
    'Tomato_Healthy', 'Tomato_Leaf Blight', 'Tomato_Leaf Curl', 'Tomato_Septoria Leaf Spot', 'Tomato_Verticillium Wilt',
    'Tomato_healthy', 'Tomato_leaf blight', 'Tomato_leaf curl', 'Tomato_septoria leaf spot', 'Tomato_verticillium wilt'
]


page = st.radio("", ["Home", "Classifier"])

# --- MAIN CONTENT ---
if page == "Home":
    st.title("Crop Pest and Disease Classifier")
    st.header("About")
    st.write("""
    This app predicts crop pests and diseases based on images you provide.

    **Detectable pests and diseases include:**
    - **Cashew:** anthracnose, gumosis, leafminer, red rust
    - **Cassava:** bacterial blight, brown spot, green mite, mosaic
    - **Maize:** fall armyworm, grasshopper, leaf beetle, leaf blight, leaf spot, streak virus
    - **Tomato:** leaf blight, leaf curl, septoria leaf spot, verticillium wilt
    """)
    st.info("Use the sidebar to navigate to the classifier page and upload your crop image.")

elif page == "Classifier":
    st.title("Crop Pest and Disease Classifier")
    st.write("Upload a crop image or take a photo to detect any pest or disease affecting it.")

    # Option to upload or take a picture
    option = st.radio("Choose input method:", ('Upload Image', 'Take Photo'))

    uploaded_file = None
    if option == 'Upload Image':
        uploaded_file = st.file_uploader("Choose an image of the crop", type=["jpg", "jpeg", "png"])
    elif option == 'Take Photo':
        uploaded_file = st.camera_input("Take a picture")

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_container_width=True)

        # Preprocess the image
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)

        if predicted_index < len(CLASS_NAMES):
            predicted_label = CLASS_NAMES[predicted_index]
            confidence = prediction[0][predicted_index] * 100

            # Display prediction
            st.success(f"**Prediction:** {predicted_label}")
            st.info(f"**Confidence:** {confidence:.2f}%")
        else:
            st.error("Prediction index out of range. Check your model and class names.")
