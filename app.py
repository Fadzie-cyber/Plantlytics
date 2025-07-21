import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('model3.h5')

# Define your class labels
CLASS_NAMES = [
    # ... (your CLASS_NAMES list)
]

page = st.radio("", ["Home", "Classifier"])

if page == "Home":
    st.title("Crop Pest and Disease Classifier")
    # ... (rest of the home page)

elif page == "Classifier":
    st.title("Crop Pest and Disease Classifier")
    st.write("Upload a crop image or take a photo to detect any pest or disease affecting it.")

    option = st.radio("Choose input method:", ('Upload Image', 'Take Photo'))

    uploaded_file = None
    if option == 'Upload Image':
        uploaded_file = st.file_uploader("Choose an image of the crop", type=["jpg", "jpeg", "png"])
    elif option == 'Take Photo':
        uploaded_file = st.camera_input("Take a picture")

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert('RGB')

            # Convert image to array for Streamlit display to avoid keyword argument issue
            img_array_for_display = np.array(img)
            st.image(img_array_for_display, caption='Uploaded Image')

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

                st.success(f"**Prediction:** {predicted_label}")
                st.info(f"**Confidence:** {confidence:.2f}%")
            else:
                st.error("Prediction index out of range. Check your model and class names.")

        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")
    else:
        st.warning("Please upload or capture an image to proceed.")
