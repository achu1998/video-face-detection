import cv2
import torch
import numpy as np
import sys
from facenet_pytorch import MTCNN
from deepface import DeepFace

# Initialize MTCNN (Face Detector)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# Load the video
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

# Age classification threshold
CHILD_AGE_THRESHOLD = 25  # Blur faces predicted to be below this age

# Process video frame by frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    progress = (frame_count / total_frames) * 100  # Compute percentage completed

    # Convert frame to RGB (MTCNN expects RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for (x1, y1, x2, y2) in boxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extract face ROI
            face = frame[y1:y2, x1:x2]
            
            if face is None or face.size == 0:
                # Print progress and age on the same line
                sys.stdout.write(f"\rProcessing... {progress:.2f}% | Detected Age: No Face Detected!")
                sys.stdout.flush()
                continue

            # Convert face to proper format for DeepFace
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            try:
                # Predict age
                analysis = DeepFace.analyze(face_rgb, actions=["age"], enforce_detection=False)
                predicted_age = analysis[0]["age"]

                # Print progress and age on the same line
                sys.stdout.write(f"\rProcessing... {progress:.2f}% | Detected Age: {predicted_age}")
                sys.stdout.flush()

                # If the detected age is below CHILD_AGE_THRESHOLD, blur the face
                if predicted_age < CHILD_AGE_THRESHOLD:
                    blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
                    frame[y1:y2, x1:x2] = blurred_face  # Replace original face with blurred face

            except Exception as e:
                sys.stdout.write(f"\rProcessing... {progress:.2f}% | Age estimation failed")
                sys.stdout.flush()

    # Write processed frame to video
    out.write(frame)

# Final print statement to avoid overwriting output
print("\nProcessing complete!")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
