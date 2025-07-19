import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
import os
from io import BytesIO

# -------------------- Configuration --------------------
st.set_page_config(page_title="YOLOv8 Detection App", layout="centered")
st.title("🧠 YOLOv8 Sign Board Detection")
st.markdown("Upload an **image**, **video**, or use **live camera** to detect objects using your trained YOLOv8 model.")

# -------------------- Load Model --------------------
MODEL_PATH = "trainedmodel.pt"  # Update if needed
model = YOLO(MODEL_PATH)

# -------------------- Sidebar --------------------
option = st.sidebar.radio("Choose Input Type", ["Image", "Video", "Webcam"])

# -------------------- Image Upload --------------------
if option == "Image":
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = uploaded_file.read()
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        st.image(image, caption="📷 Original Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        with st.spinner("🔍 Running detection..."):
            results = model(tmp_path)
            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(annotated_rgb)

        st.success("✅ Detection complete!")
        st.image(result_image, caption="📌 Detected Image", use_container_width=True)

        # Download button
        img_byte_arr = BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        st.download_button(
            label="📥 Download Result Image",
            data=img_byte_arr.getvalue(),
            file_name="detected_image.png",
            mime="image/png"
        )

        os.remove(tmp_path)

# -------------------- Video Upload --------------------
elif option == "Video":
    uploaded_video = st.file_uploader("📤 Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(uploaded_video)

        run_vid = st.button("🚀 Run Detection on Video")

        if run_vid:
            stframe = st.empty()
            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)
                annotated_frame = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_rgb, channels="RGB", use_container_width=True)

            cap.release()
            os.remove(video_path)

# -------------------- Webcam --------------------
elif option == "Webcam":
    start = st.button("🎥 Start Webcam")
    stop = st.button("⏹️ Stop Webcam")

    if start and not stop:
        st.warning("Press **⏹️ Stop Webcam** to end feed.")
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Failed to grab frame.")
                break

            results = model(frame)
            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_rgb, channels="RGB", use_container_width=True)

            # Stop condition
            if st.session_state.get("stop_webcam", False):
                break

        cap.release()

    if stop:
        st.session_state["stop_webcam"] = True

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("Created by SAKET , PAYOLI , SIMRAN")
st.markdown("Made using [YOLOv8](https://github.com/ultralytics/ultralytics) and [Streamlit](https://streamlit.io)")
