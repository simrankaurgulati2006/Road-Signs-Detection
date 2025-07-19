import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading

# Load YOLOv8 model
model = YOLO("trainedmodel.pt")

# Global state
cap = None
running = False
thread = None

# Initialize window
root = tk.Tk()
root.title("YOLOv8 Real-Time Detection")
root.geometry("1000x700")

# Display label for image/video
display_label = tk.Label(root)
display_label.pack()

# ---------------------- Functions ---------------------- #
def stop_video():
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None
        display_label.config(image='')

def show_frame():
    global cap, running
    
    while running and cap:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for performance
        frame = cv2.resize(frame, (640, 360))

        # Run detection
        results = model(frame, verbose=False)
        annotated = results[0].plot()

        # Convert to Tkinter format
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(annotated_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Display in label
        display_label.imgtk = imgtk
        display_label.configure(image=imgtk)

        # Force small delay for GUI update
        display_label.update_idletasks()

    stop_video()

def start_thread():
    global thread
    thread = threading.Thread(target=show_frame)
    thread.start()

def browse_video():
    global cap, running
    stop_video()
    path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if path:
        cap = cv2.VideoCapture(path)
        running = True
        start_thread()

def start_webcam():
    global cap, running
    stop_video()
    cap = cv2.VideoCapture(0)
    running = True
    start_thread()

def browse_image():
    stop_video()
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not path:
        return

    frame = cv2.imread(path)
    frame = cv2.resize(frame, (640, 360))
    results = model(frame, verbose=False)
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(annotated_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    display_label.imgtk = imgtk
    display_label.configure(image=imgtk)

# ---------------------- Buttons ---------------------- #
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Choose Image", command=browse_image, width=15).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="Choose Video", command=browse_video, width=15).grid(row=0, column=1, padx=10)
tk.Button(btn_frame, text="Start Webcam", command=start_webcam, width=15).grid(row=0, column=2, padx=10)
tk.Button(btn_frame, text="Stop", command=stop_video, width=15).grid(row=0, column=3, padx=10)

# ---------------------- Exit Cleanup ---------------------- #
def on_closing():
    stop_video()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
