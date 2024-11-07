import os
import cv2
import zipfile
import tempfile
import numpy as np
import streamlit as st
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.svm import SVC
import time

# Initialize FaceNet and MTCNN detector
embedder = FaceNet()
detector = MTCNN()

# Initialize session state variables
if "capture_count" not in st.session_state:
    st.session_state.capture_count = 0
if "captured_images" not in st.session_state:
    st.session_state.captured_images = []
if "surrounding_stage" not in st.session_state:
    st.session_state.surrounding_stage = 1

surroundings_instructions = [
    "natural lighting",
    "dim lighting",
    "dim lighting with camera flash",
    "wearing any accessories you might wear (e.g., glasses, hat, etc.)",
    "no smile or different smiles"
]

# Display app title and instructions
st.title("Face Recognition Filter")
st.write("Capture 100 photos of yourself for training in different surroundings, then upload a folder of images to filter those with your face.")

# Step 1: Capture 100 photos from webcam in intervals of 20
if st.session_state.surrounding_stage <= len(surroundings_instructions):
    st.write(f"Step 1: Capture Photos for Training")
    st.write(f"Current Surrounding: {surroundings_instructions[st.session_state.surrounding_stage - 1]}")
else:
    st.write("All 100 photos captured. Proceed to the next step.")

# Button to start the capture process
if st.button("Capture 20 Photos"):
    # Check if we're still within the 100-photo limit
    if st.session_state.capture_count < 100:
        st.write("Please wait, capturing 20 photos...")

        for i in range(20):
            # Prompt the user to capture each photo one-by-one
            img_file = st.camera_input("Capture photo", key=f"capture_{st.session_state.capture_count + i}")

            if img_file is not None:
                # Convert the photo to a usable format and store in session state
                st.session_state.captured_images.append(img_file)
                st.session_state.capture_count += 1
                st.write(f"Captured {st.session_state.capture_count}/100 photos.")
        
        # Update surroundings after each batch of 20 photos
        if st.session_state.capture_count % 20 == 0:
            st.session_state.surrounding_stage += 1
            if st.session_state.surrounding_stage <= len(surroundings_instructions):
                st.write(f"Please change your surroundings to: {surroundings_instructions[st.session_state.surrounding_stage - 1]}")
            else:
                st.write("All 100 photos captured. Proceed to the next step.")
    else:
        st.write("You have captured all 100 photos. Proceed to the next step.")
st.write(f"capture count: {st.session_state.capture_count}")
# Step 2: Generate embeddings for captured images and train SVM
if st.session_state.capture_count == 100:
    face_embeddings = []
    labels = []

    for img_file in st.session_state.captured_images:
        image = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
        detections = detector.detect_faces(image)

        for detection in detections:
            x, y, width, height = detection['box']
            face = image[y:y+height, x:x+width]
            face_resized = cv2.resize(face, (160, 160))
            face_normalized = face_resized.astype('float32') / 255.0
            embedding = embedder.embeddings(np.expand_dims(face_normalized, axis=0))[0]
            face_embeddings.append(embedding)
            labels.append(1)  # Label as "1" for your face

    face_embeddings = np.array(face_embeddings)
    labels = np.array(labels)

    # Train SVM model on embeddings
    svm = SVC(kernel='linear', probability=True)
    svm.fit(face_embeddings, labels)
    st.success("SVM model trained successfully!")

uploaded_file = st.file_uploader("Upload a folder of images (zip file)", type="zip")

if uploaded_file and st.session_state.capture_count == 100:
    if st.button("Start Filtering"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_folder = os.path.join(tmp_dir, "input_folder")
            output_folder = os.path.join(tmp_dir, "output_folder")
            
            with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                zip_ref.extractall(input_folder)

            os.makedirs(output_folder, exist_ok=True)
            for image_name in os.listdir(input_folder):
                image_path = os.path.join(input_folder, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    continue

                detections = detector.detect_faces(image)
                face_found = False

                for detection in detections:
                    x, y, width, height = detection['box']
                    face = image[y:y+height, x:x+width]

                    embedding = embedder.embeddings(np.expand_dims(face, axis=0))

                    prediction = svm.predict(embedding)
                    if prediction == 1:
                        face_found = True
                        break

                if face_found:
                    cv2.imwrite(os.path.join(output_folder, image_name), image)

            output_zip_path = os.path.join(tmp_dir, "filtered_images.zip")
            with zipfile.ZipFile(output_zip_path, "w") as zip_out:
                for root, dirs, files in os.walk(output_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zip_out.write(file_path, os.path.relpath(file_path, output_folder))

            with open(output_zip_path, "rb") as f:
                st.download_button(
                    label="Download Filtered Images",
                    data=f,
                    file_name="filtered_images.zip",
                    mime="application/zip"
                )

            st.success("Filtering complete. Click the button above to download images with your face.")

