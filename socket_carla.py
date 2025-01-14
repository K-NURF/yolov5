import socket
import pickle
import torch
import numpy as np
import cv2
import os
from datetime import datetime
from utils.general import non_max_suppression, scale_boxes, increment_path
from utils.plots import Annotator, colors
from models.common import DetectMultiBackend

# Directories
OUTPUT_DIR = "annotated_frames"
VIDEO_DIR = "videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# Model weights and device
weights_path = 'yolov5/runs/train/exp2/weights/best.pt'  # Replace with your model path
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

# Load YOLOv5 model
print("[INFO] Loading YOLOv5 model...")
model = DetectMultiBackend(weights_path, device=device)
model.conf = 0.5  # Confidence threshold
model.iou = 0.45  # IoU threshold
model.classes = None  # Detect all classes
names = model.names  # Class names
print("[INFO] YOLOv5 model loaded successfully.")

def clear_annotated_frames(directory):
    """Delete all files in the specified directory."""
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Remove file
            except Exception as e:
                print(f"[ERROR] Failed to delete {file_path}: {e}")
    else:
        os.makedirs(directory)  # Create directory if it doesn't exist


def annotate_frame(frame, detections, img_size):
    """Annotate the frame with bounding boxes and labels."""
    annotator = Annotator(frame, line_width=2, example=str(names))
    if detections is not None:
        detections[:, :4] = scale_boxes(img_size, detections[:, :4], frame.shape).round()  # Rescale boxes
        for *xyxy, conf, cls in detections:
            label = f"{names[int(cls)]} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(int(cls), True))
    return annotator.result()

def create_video_from_frames(input_folder, output_file, frame_rate):
    """Create a video from saved frames."""
    frames = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg'))]
    frames.sort()  # Ensure frames are sorted
    if not frames:
        print("[ERROR] No frames found in the directory.")
        return

    # Read the first frame to get dimensions
    first_frame_path = os.path.join(input_folder, frames[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape

    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

    for frame_name in frames:
        frame_path = os.path.join(input_folder, frame_name)
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()
    print(f"[INFO] Video saved to {output_file}")

def main():
    # Connect to CARLA server
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect(('172.30.0.1', 12345))  # Replace with your server's IP/host
        print("[INFO] Connected to the CARLA server.")
    except ConnectionRefusedError:
        print("[ERROR] Unable to connect to the server. Ensure it's running.")
        return

    frame_count = 0  # Counter for saving frames

    clear_annotated_frames(OUTPUT_DIR)

    try:
        while True:
            # Step 1: Receive the length of the incoming frame (4 bytes)
            frame_size_data = client.recv(4)  # Read 4 bytes for the size prefix
            if len(frame_size_data) < 4:
                print("[ERROR] Failed to read frame size. Connection may be closed.")
                break

            # Decode frame size
            frame_size = int.from_bytes(frame_size_data, 'big')
            if frame_size <= 0 or frame_size > 10**7:  # Sanity check: Limit to ~10 MB
                print(f"[ERROR] Invalid frame size: {frame_size} bytes. Skipping...")
                continue
            print(f"[DEBUG] Expected frame size: {frame_size} bytes")

            # Step 2: Receive the frame data in chunks
            data = b""
            while len(data) < frame_size:
                packet = client.recv(min(4096, frame_size - len(data)))  # Adjust chunk size dynamically
                if not packet:
                    print("[ERROR] Connection closed while receiving frame.")
                    break
                data += packet
            if len(data) != frame_size:
                print(f"[ERROR] Incomplete frame received. Expected {frame_size} bytes, got {len(data)} bytes.")
                continue

            # Step 3: Deserialize and process the frame
            try:
                frame = pickle.loads(data)  # Deserialize the frame
                print(f"[DEBUG] Received frame shape: {frame.shape}")
            except Exception as e:
                print(f"[ERROR] Failed to deserialize frame: {e}")
                continue

            # Process the frame with YOLOv5
            img = cv2.resize(frame, (640, 640))  # Resize to model input size
            img_tensor = torch.from_numpy(img).to(device).float() / 255.0  # Normalize to [0, 1]
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to CHW and add batch dimension

            predictions = model(img_tensor)
            detections = non_max_suppression(predictions, model.conf, model.iou, model.classes)
            print(f"[DEBUG] Detections: {detections}")

            # Save annotated frame for debugging
            annotated_frame = annotate_frame(frame, detections[0], img_tensor.shape[2:])
            frame_count += 1
            save_path = os.path.join(OUTPUT_DIR, f"annotated_{frame_count:04d}.jpg")
            cv2.imwrite(save_path, annotated_frame)
            print(f"[INFO] Saved annotated frame: {save_path}")

    except KeyboardInterrupt:
        print("[INFO] Interrupted. Creating video from saved frames...")
        video_path = os.path.join(VIDEO_DIR, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        create_video_from_frames(OUTPUT_DIR, video_path, frame_rate=30)
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        print("[INFO] Closing connection.")
        client.close()

if __name__ == '__main__':
    main()
