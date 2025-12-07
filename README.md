📚 Sign Language Detection System

Real-Time ASL/ISL Recognition using MediaPipe & Random Forests

🧭 Overview

This project bridges communication gaps between the Deaf & Hard of Hearing (DHH) community and the hearing population.
It translates American Sign Language (ASL) and Indian Sign Language (ISL) hand gestures to text (and speech) using real-time computer vision.

Unlike expensive sensor-based gloves or GPU-dependent CNNs, this system operates fully on CPU using:

MediaPipe for 21-landmark hand tracking

Random Forest Classifier for high-accuracy predictions

🔍 Why This Approach?

Traditional systems rely on:

Flex-sensor gloves (expensive, non-portable)

Depth sensors (hardware dependent)

CNNs (heavy computation, large datasets)

This project:

Uses lightweight mathematical landmark geometry

Requires only tabular features (42 values)

Runs smoothly on low-end laptops

🧠 Core Principles
🖐 MediaPipe Hand Tracking

Detects 21 3D hand landmarks

Works in real time using BlazePalm + Landmark Model

Bounding box reused to speed future frames

🌲 Random Forest Classification

Classifies gestures based on 42 normalized coordinates

Fast CPU inference

Stable and interpretable model

🗂 Project Structure
Sign-Language-Detector/
├── data/
│   ├── A/
│   ├── B/
│   └── ...
├── models/
│   ├── data.pickle
│   └── model.p
├── static/
│   ├── css/
│   └── js/
├── templates/
│   └── index.html
├── app.py
├── collect_imgs.py
├── create_dataset.py
├── train_classifier.py
├── inference_classifier.py
├── requirements.txt
└── README.md

🛠 Technology Stack
Technology	Purpose
OpenCV	Camera input, visualization
MediaPipe	Landmark detection
Scikit-Learn	Random Forest classification
Flask	Web-based deployment
NumPy	Numerical feature manipulation
Pickle	Model/data serialization
📥 Installation
git clone https://github.com/<your-username>/sign-language-detector.git
cd sign-language-detector
pip install -r requirements.txt

📸 Phase I — Data Collection

Run:

python collect_imgs.py


You will:

Show gestures to webcam

Press Q to start automatic capture

~100 images per class recommended

⚙ Phase II — Feature Extraction

Run:

python create_dataset.py


This:

Extracts 21 landmarks

Converts them to 42-D geometry

Saves data.pickle

🤖 Phase III — Training Model

Run:

python train_classifier.py


Outputs:

Training accuracy

Stores model as model.p

🗝 Phase IV — Real-Time Detection

Run:

python inference_classifier.py


Displays:

Hand bounding box

Predicted sign label

Latency: 20–30 FPS on CPU

🌐 Flask Web Deployment

Run:

python app.py


Browser:

http://127.0.0.1:5000/

📊 How It Works — Simple Explanation

Camera feeds frames

MediaPipe extracts 21 landmarks

Coordinates are normalized

42-value vector goes to Random Forest

Forest votes → predicted sign

Display result

🚀 Performance & Insights
Why Tabularization Works

Rather than classify millions of pixels, we classify 42 precise geometry values.

This is:

Robust

Lightweight

Deployable anywhere

Latency Optimization

Bounding box reused

Model inference cost < 1 ms

Smooth webcam rendering

🧭 Limitations & Future Enhancements
Planned Improvements:

✔ Dynamic sign recognition (LSTMs/GRUs)
✔ 2-hand gesture support
✔ NLP auto-correction
✔ Mobile edge deployment (ONNX/TFLite)

🥇 Conclusion

This project proves that real-time sign recognition does not require deep learning or GPUs.

It brings:

Accuracy

Accessibility

Deployability

A meaningful step toward inclusive communication.

📬 Contribution

Pull Requests welcome!
If you’d like dataset access or improved models, open an issue.

📜 License

MIT License.

👤 Author

Niteesh Pandit
