import streamlit as st
import pandas as pd
import cv2
import os
import torch
from datetime import datetime
from pathlib import Path
from utils.general import check_img_size, increment_path, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors  # Import Annotator and colors for color-coded bounding boxes
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

# Constants
OUTPUT_DIR = "output_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = "runs/train/exp2/weights/best.pt"  # Replace with your YOLOv5 model path

# Device Selection
device = select_device("")  # Automatically selects the best device (GPU or CPU)
print(f"[INFO] Using device: {device}")

# Load YOLOv5 Model
def load_yolov5_model(weights, device):
    try:
        print("[INFO] Loading YOLOv5 model...")
        model = DetectMultiBackend(weights, device=device)
        model.warmup(imgsz=(1, 3, 640, 640))  # Warm-up model with sample input
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        st.stop()

model = load_yolov5_model(MODEL_PATH, device)
names = model.names  # Class names
stride = model.stride  # Model stride

# Function to Annotate Frames
def annotate_frame(frame, detections, original_shape, img_shape):
    """Annotate the frame with bounding boxes and labels."""
    annotator = Annotator(frame, line_width=2, example=str(names))
    detections[:, :4] = scale_boxes(img_shape, detections[:, :4], original_shape).round()  # Scale boxes to original
    for *xyxy, conf, cls in detections:
        label = f"{names[int(cls)]} {conf:.2f}"
        annotator.box_label(xyxy, label, color=colors(int(cls), True))  # Color-coded bounding boxes
    return annotator.result()

# Process Video Function
def process_video(uploaded_video):
    """Process the uploaded video, run YOLOv5 detection, and save output."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file_path = f"temp_video_{timestamp}.mp4"

    # Save the uploaded video
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_video.read())

    # Display the uploaded video
    st.video(temp_file_path)

    # Setup video processing
    cap = cv2.VideoCapture(temp_file_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_file_path = os.path.join(OUTPUT_DIR, f"processed_video_{timestamp}.mp4")
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

    st.write("Processing the video...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        img = cv2.resize(frame, (640, 640))  # Resize to model input size
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to CHW + batch dimension

        # Run YOLOv5 inference
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45, max_det=1000)[0]

        # Annotate the frame if detections exist
        if pred is not None and len(pred):
            annotated_frame = annotate_frame(frame, pred, (height, width), img_tensor.shape[2:])
        else:
            annotated_frame = frame

        # Write to the output video
        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()

    st.success(f"Video processing complete! Processed {frame_count} frames.")
    st.video(output_file_path, format="video/mp4")
    st.write(f"Processed video saved to: {output_file_path}")

# Streamlit Application
st.title("Object Detection Using Computer Vision through Transfer Learning for Autonomous Driving")

menu = ["Overview", "Video Processing"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Overview":
    st.header("Overview")
    st.write("This application demonstrates object detection using a fine-tuned YOLOv5 model.")
    st.write("The goal was to develop and validate a transfer learning-based model for accurate object detection and classification in diverse driving environments to enhance the performance and reliability of autonomous driving systems")
    st.write("You can upload videos and view results with bounding boxes and labels.")

elif choice == "Video Processing":
    st.header("Upload and Process Video")
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_video:
        process_video(uploaded_video)
