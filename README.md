# 📚 Sign Language Detection System (Deep Learning Upgrade)

Real-Time ASL/ISL Recognition using YOLO26s & Flask.

## 🧭 Overview

This project bridges communication gaps between the Deaf & Hard of Hearing (DHH) community and the hearing population. It translates American Sign Language (ASL) and Indian Sign Language (ISL) hand gestures to text using real-time computer vision.

**Version Update:** Previously built using Machine Learning (MediaPipe and Random Forests), this project has been fully upgraded to a **Deep Learning** pipeline using **YOLO26s**. 

## 🔍 Why Deep Learning (YOLO)?

The previous approach required extracting 21 hand landmarks and passing 42 normalized coordinates to a Random Forest Classifier. While lightweight, traditional tabularization can be limited by complex backgrounds, varied lighting, and gesture occlusions.

The new Deep Learning approach uses YOLO26s:
* **End-to-End Inference:** YOLO26s is NMS-free (end-to-end), meaning it requires no post-processing steps.
* **Unified Pipeline:** The Ultralytics API seamlessly loads the model to perform direct predictions on raw pixels.
* **Robustness:** The model learns spatial features directly from the images, relying on a large, restructured dataset of 20,274 images across train, valid, and test splits.

## 🗂 Project Structure

\`\`\`text
Sign-Language-Detector/
├── American-sign-language-2/   # Restructured YOLO flat dataset
│   ├── images/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── valid/
│       └── test/
├── models/
│   └── best.pt                 # Trained YOLO26s model weights
├── templates/
│   └── index.html              # Frontend detection interface
├── static/
│   ├── css/style.css           # UI Styling
│   └── js/script.js            # HUD and system logic
├── app.py                      # Flask backend and video streaming
└── requirements.txt            # Project dependencies
\`\`\`

## 🤖 Phase I — Dataset Preparation

The dataset was originally downloaded in a Roboflow structure and has been restructured into a YOLO flat structure (`images/train`, `labels/train`, etc.) for seamless Ultralytics integration.

## 🤖 Phase II — Training the Model

Train the YOLO model on the restructured dataset. Once training is completed, the inference weights are saved (e.g., `best.pt`) and must be placed in the project root or a `models/` folder.

## 🌐 Phase III — Web Deployment (Flask)

Unlike the previous local OpenCV window, the Deep Learning version features a fully integrated web interface.

1. **Install dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
   *(Requires: `flask`, `opencv-python`, and `ultralytics`)*.

2. **Run the Flask server:**
   \`\`\`bash
   python app.py
   \`\`\`

3. **Open your browser:**
   \`\`\`text
   http://127.0.0.1:5000/
   \`\`\`

## 📊 How It Works — Deep Learning Pipeline

1. The browser interface connects to the Flask backend to request the `/video_feed`.
2. `app.py` captures webcam frames using OpenCV.
3. Frames are passed directly to the YOLO26s model for real-time inference.
4. Predictions (with a Confidence Threshold > 0.50) are overlaid as text directly on the annotated frame.
5. The annotated frame is encoded to JPEG and streamed continuously back to the browser via `multipart/x-mixed-replace`.

## 🚀 Performance & Insights

* **NMS-Free Efficiency:** Because the model lacks Non-Maximum Suppression overhead, inference is highly optimized for real-time video feeds.
* **Modern UI:** The frontend includes a "NeuroSign" Detection Interface featuring live diagnostics, system logs, latency stats, and confidence tracking.

## 🧭 Future Enhancements
* Incorporate recurrent layers (LSTMs/GRUs) for dynamic sign recognition.
* Two-hand gesture support and NLP auto-correction.
