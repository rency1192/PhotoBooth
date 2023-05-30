import base64
import os
import face_recognition
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import shutil

# Step 1: Create a Streamlit web app
st.title("Face Matching Web App")

# Step 2: Define the path to the dataset directory
dataset_dir = "E:/InnovaticsInternship/DataSet"

output_dir = "E:/InnovaticsInternship/MatchedPhotos"

# Step 3: Get the list of image files in the dataset directory
image_files = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.endswith(".jpg")]

# Step 4: Function to perform face matching
def perform_face_matching(query_image):
    # Load the query image
    query_image_data = face_recognition.load_image_file(query_image)
    query_face_encodings = face_recognition.face_encodings(query_image_data)

    # Iterate over the dataset images and compare with the query image
    matched_images = []
    for image_file in image_files:
        image_data = face_recognition.load_image_file(image_file)
        image_face_encodings = face_recognition.face_encodings(image_data)
        for query_face_encoding in query_face_encodings:
            matches = face_recognition.compare_faces(image_face_encodings, query_face_encoding)
            if True in matches:
                matched_images.append(image_file)
                break

    return matched_images

# Step 5: Add instructions and upload button
st.write("Scan your face and click the button to find matching photos.")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Step 6: Perform face matching and display matched photos
if uploaded_image is not None:
    matched_photos = perform_face_matching(uploaded_image)
    if matched_photos:
        for i, photo in enumerate(matched_photos):
            image = Image.open(photo)
            st.image(image, caption=os.path.basename(photo), use_column_width=True)

            # Display download button
            download_button = st.button(f"Download {i + 1}")
            if download_button:
                with open(photo, "rb") as f:
                    bytes_data = f.read()
                b64_data = base64.b64encode(bytes_data).decode()
                href = f'<a href="data:image/jpg;base64,{b64_data}" download="{os.path.basename(photo)}">Download {os.path.basename(photo)}</a>'
                st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("No matching photos found.")

st.markdown(
    """
    <style>
    body {
        background-color: #F0F4F6;
        color: #66d9c0;
    }
    h1 {
        color: #2068ba;
        text-align: center;
    }
    .stButton button {
        background-color: #FFC300;
        color: #FFFFFF;
    }
    .my-text {
        font-size: 18px;
        color: #ab1149;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)