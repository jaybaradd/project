import os
import cv2
import zipfile
import shutil
import tempfile
import numpy as np
import streamlit as st
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.svm import SVC
import joblib  # Assuming your SVM model is saved using joblib

# Load FaceNet model and MTCNN face detector
embedder = FaceNet()
detector = MTCNN()

# Load your pre-trained SVM model
svm = joblib.load("/Users/rushikesh/Desktop/Jay/vscode_foml/project/svm_model.pkl")  # Update with your SVM model path

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        st.write(f"Processing image: {image_name}")
        detections = detector.detect_faces(image)
        face_found = False

        for detection in detections:
            x, y, width, height = detection['box']
            face = image[y:y+height, x:x+width]  # Crop the face region

            # Generate embedding for the cropped face
            embedding = embedder.embeddings(np.expand_dims(face, axis=0))

            # Use SVM to predict if this is your face
            prediction = svm.predict(embedding)
            if prediction == 1:
                face_found = True
                break

        # Save image if face is found
        if face_found:
            cv2.imwrite(os.path.join(output_folder, image_name), image)
            st.write(f"Face found in {image_name}")

# Streamlit App
st.title("Face Recognition Filter")
st.write("Upload a folder of images. The app will filter images with your face, allowing you to download the results.")

# Upload images as a zip file
uploaded_file = st.file_uploader("Upload a folder of images (zip file)", type="zip")

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_folder = os.path.join(tmp_dir, "input_folder")
        output_folder = os.path.join(tmp_dir, "output_folder")
        
        # Unzip uploaded file
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            zip_ref.extractall(input_folder)

        # Process images
        process_images(input_folder, output_folder)

        # Zip processed output
        output_zip_path = os.path.join(tmp_dir, "filtered_images.zip")
        with zipfile.ZipFile(output_zip_path, "w") as zip_out:
            for root, dirs, files in os.walk(output_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_out.write(file_path, os.path.relpath(file_path, output_folder))

        # Download link
        with open(output_zip_path, "rb") as f:
            st.download_button(
                label="Download Filtered Images",
                data=f,
                file_name="filtered_images.zip",
                mime="application/zip"
            )

        st.success("Processing complete. Click the button above to download images with your face.")