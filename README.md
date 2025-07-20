# Plantlytics
This project leverages Machine Learning to predict crop pests and diseases from images of crops. The goal is to assist farmers in identifying the specific pests or diseases affecting their crops early. By using image-based predictions, this tool aims to support sustainable agriculture and improve crop yields.


This is a Streamlit-based web application that classifies images of crops to detect possible pests and diseases using a trained deep learning model.

🚀 Features
Detects pests and diseases affecting:

Cashew: Anthracnose, Gumosis, Leafminer, Red Rust

Cassava: Bacterial Blight, Brown Spot, Green Mite, Mosaic

Maize: Fall Armyworm, Grasshopper, Leaf Beetle, Leaf Blight, Leaf Spot, Streak Virus

Tomato: Leaf Blight, Leaf Curl, Septoria Leaf Spot, Verticillium Wilt

Accepts images via:

Upload from device

Direct camera capture

Provides:

Prediction label for the identified disease or pest

Confidence score for the prediction

A simple sidebar for navigation with a logo.

📁 Project Structure
bash
Copy
Edit
plantlytics/
│
├── app.py              # Main Streamlit app script
├── model3.h5           # Pre-trained Keras model
├── logo.png            # Logo image for the sidebar
├── requirements.txt    # List of dependencies (optional but recommended)
└── README.md           # Project documentation

🛠️ Setup Instructions
1. Clone the Repository

git clone <repository-url>
cd plantlytics
2. Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate    # On Linux/macOS
.\venv\Scripts\activate     # On Windows
3. Install Dependencies

pip install streamlit tensorflow pillow numpy
Optionally, list these in a requirements.txt and install with:


pip install -r requirements.txt
4. Run the App

streamlit run app.py
