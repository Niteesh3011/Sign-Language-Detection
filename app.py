import cv2
import numpy as np
import pickle
import os
import mediapipe as mp
from flask import Flask, render_template, Response

app = Flask(__name__)

# --- 1. Robust Model Loading ---
model_path = './model.p'
if not os.path.exists(model_path):
    # Check alternate location
    model_path = './models/model.p'

model = None
try:
    if os.path.exists(model_path):
        model_dict = pickle.load(open(model_path, 'rb'))
        model = model_dict['model']
        print(f"✅ Model loaded successfully from: {model_path}")
    else:
        print("⚠️ Model file not found! Prediction will be disabled.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- 2. Setup MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# --- 3. Labels Dictionary (Optional but Recommended) ---
# If your model predicts numbers (0, 1, 2), map them to letters here.
# If your model predicts letters directly, this won't be used.
labels_dict = {0: 'A', 1: 'B', 2: 'C'} 

def gen_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip frame for mirror effect (easier for user)
        # frame = cv2.flip(frame, 1)

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

                # --- PREDICTION LOGIC ---
                if model:
                    try:
                        # Make prediction
                        prediction = model.predict([np.asarray(data_aux)])
                        
                        # Handle the output
                        pred_val = prediction[0]
                        
                        # If model returns a number, try to map it, otherwise use as string
                        if isinstance(pred_val, (int, np.integer)) and pred_val in labels_dict:
                            predicted_character = labels_dict[pred_val]
                        else:
                            predicted_character = str(pred_val)

                        # Draw Box & Text
                        x1 = int(min(x_) * W) - 10
                        y1 = int(min(y_) * H) - 10
                        x2 = int(max(x_) * W) - 10
                        y2 = int(max(y_) * H) - 10

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    except Exception as e:
                        # This prints errors to your terminal if shape mismatch happens
                        print(f"Prediction Error: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)