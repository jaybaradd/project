import os
import tempfile
import zipfile
import cv2
import numpy as np
import streamlit as st
from mtcnn import MTCNN
import joblib
from keras_facenet import FaceNet  # Ensure you have this imported correctly

embedder = FaceNet()
detector = MTCNN()
svm = joblib.load("/Users/rushikesh/Desktop/Jay/vscode_foml/project/svm_model.pkl")  # Update with your SVM model path

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    saved_images_count = 0  # Track the number of saved images
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        # st.write(f"Processing image: {image_name}")
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
            output_image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_image_path, image)
            # st.write(f"Face found in {image_name}. Saved to {output_image_path}")
            saved_images_count += 1

    # Check if any images were saved
    if saved_images_count == 0:
        st.warning("No images with detected faces were saved.")
    else:
        st.write(f"Total saved images: {saved_images_count}")

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

        # Check if any images were saved before creating the zip
        if len(os.listdir(output_folder)) > 0:
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
        else:
            st.warning("No images were saved in the output folder, so the zip file is empty.")
