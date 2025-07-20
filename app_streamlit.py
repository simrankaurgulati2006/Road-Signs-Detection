import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
import os

# Load YOLO model
model = YOLO("trainedmodel.pt")

st.title("🚦 Road Sign Detection ")
st.write("Upload an image or video to detect road signs using YOLOv8.")

# Image Upload
st.header("Image Detection")

uploaded_image = st.file_uploader("Choose an Image", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    img = Image.open(uploaded_image).convert('RGB')
    img_np = np.array(img)

    results = model(img_np)
    annotated = results[0].plot()

    st.image(annotated, caption="Detected Road Signs", use_container_width=True)

# Video Upload
st.header("Video Detection")

uploaded_video = st.file_uploader("Choose a Video", type=['mp4', 'avi', 'mov'])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        results = model(frame)
        annotated = results[0].plot()

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_rgb, channels="RGB", use_container_width=True)

    cap.release()
    os.remove(tfile.name)
