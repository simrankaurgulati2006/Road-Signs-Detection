import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained YOLOv8 model
model = YOLO('trainedmodel.pt')  # Replace with your model's path

# Function to upload and detect
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if file_path:
        # Run YOLOv8 detection
        results = model(file_path)

        # Plot results and convert to image
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(annotated_frame)
        img.thumbnail((600, 600))  # Resize if too large
        imgtk = ImageTk.PhotoImage(img)

        # Update the image label
        image_label.config(image=imgtk)
        image_label.image = imgtk

# Setup Tkinter window
root = tk.Tk()
root.title("YOLOv8 Image Detection")
root.geometry("700x700")

btn = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 14), bg="lightblue")
btn.pack(pady=20)

image_label = tk.Label(root)
image_label.pack()

root.mainloop()
