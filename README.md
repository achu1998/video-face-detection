# 🎥 Video Face Detection

## 📌 Overview

This project processes a video file frame by frame, detects faces using MTCNN, estimates their age using DeepFace, and applies a blur effect to faces predicted to be below a specified age threshold. The processed video is then saved as an output file.

## ✨ Features

- 🖼️ **Face Detection:** Uses **MTCNN (FaceNet-PyTorch)** for accurate face detection.
- 📊 **Age Estimation:** Implements **DeepFace** to estimate the age of detected faces.
- 🛡️ **Privacy Protection:** Blurs faces of individuals below a specified age threshold.
- 📢 **Real-Time Processing:** Displays progress and detected age in the console.
- 🎥 **Video Output:** Saves the processed video as an output file.

## 📦 Requirements

First, navigate to the project directory:

```sh
cd video-face-detection
```

Then, install the required dependencies:

```sh
pip install -r requirements.txt
```


## 🚀 Usage
1. Place the video file you want to process in the project directory and rename it to **input.mp4**.
2. Run the script:
   ```sh
   python3 script.py
   ```
3. The processed video will be saved as **output.mp4** in the same directory.

## ⚙️ Configuration

- The threshold age for blurring faces is set to **25** (modifiable via `CHILD_AGE_THRESHOLD`).
- The script automatically selects **CUDA** if available; otherwise, it runs on **CPU**.

## 🔍 Notes

- The script handles errors gracefully and continues processing even if face detection or age estimation fails.
- The progress and detected age are displayed in real time in the console.

------------------------------------------------------------------------

## Demo Video

Watch the demo of the application:

Input -

[![Demo Video](in.png)](https://www.youtube.com/shorts/9AeG4a7mV38)

Output -

[![Demo Video](out.png)](https://www.youtube.com/shorts/pbauoC6d6Q8)

------------------------------------------------------------------------

Developed by Ashween R

