import shutil
import streamlit as st
import cv2
import face_recognition
import os
import base64
from PIL import Image
import urllib.parse

# Step 1: Create a Streamlit web app
st.set_page_config(page_title="Face Matching Web App", page_icon=":smiley:")
st.title("Face Matching Web App")

# Step 2: Define the path to the dataset directory
dataset_dir = "E:/InnovaticsInternship/DataSet"
output_dir="E:/InnovaticsInternship/MatchedPhotos"

# Step 3: Get the list of image files in the dataset directory
image_files = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.endswith(".jpg")]

# Step 4: Function to perform face matching
def perform_face_matching(query_image):
    # Load the query image
    query_image_data = query_image
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

# Step 5: Add instructions and start the webcam video capture
st.markdown("---")
st.markdown("## Instructions")
st.write("Scan your face using the webcam and click the 'Find Matches' button to find matching photos.")
st.markdown("---")

# Step 6: Perform face matching and display matched photos
col_scan, col_result = st.columns(2)

with col_scan:
    st.header("Scan Face")
    find_matches_button = st.button("Find Matches")

    if find_matches_button:
        video_capture = cv2.VideoCapture(0)
        _, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        query_image = Image.fromarray(rgb_frame)
        st.image(query_image, channels="RGB", use_column_width=True)
        matched_photos = perform_face_matching(rgb_frame)
        st.success(f"Found {len(matched_photos)} matching photo(s)")
        if matched_photos:
            st.header("Matched Photos")
            for photo in matched_photos:
                image = Image.open(photo)
                st.image(image, caption=os.path.basename(photo), use_column_width=True)

                # Display download button
                # Download matched photo
                download_path = os.path.join(output_dir, os.path.basename(photo))
                #st.markdown(f'[Download {os.path.basename(photo)}](data:file/{os.path.basename(photo)}" download="{download_path}")',unsafe_allow_html=True)
                shutil.copy(photo, download_path)

        # Release the video capture and close the OpenCV window
        video_capture.release()
        cv2.destroyAllWindows()


with col_result:
    st.image("E:\InnovaticsInternship\MatchedPhotos", width=200)
