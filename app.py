import cv2
import os
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# ─────────────────────────────────────────────
# 1. Model Loading  (YOLO26s)
# ─────────────────────────────────────────────
# YOLO26 is NMS-free (end-to-end), so no post-processing step needed.
# The Ultralytics API is identical to YOLOv8/YOLO11 — just load and predict.

MODEL_CANDIDATES = [
    'best.pt',
    os.path.join('models', 'best.pt'),
    os.path.join('weights', 'best.pt'),
]

model = None
for candidate in MODEL_CANDIDATES:
    if os.path.exists(candidate):
        try:
            model = YOLO(candidate)
            print(f"✅ YOLO26s model loaded successfully from: {candidate}")
        except Exception as e:
            print(f"❌ Error loading model from {candidate}: {e}")
        break

if model is None:
    print("⚠️  No model file found. Place best.pt in the project root or a 'models/' folder.")


# ─────────────────────────────────────────────
# 2. Configuration
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5   # Minimum confidence to show a prediction
BOX_COLOR           = (0, 255, 0)   # Green bounding boxes
TEXT_COLOR          = (0, 255, 0)
BOX_THICKNESS       = 2
FONT                = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE          = 0.8
FONT_THICKNESS      = 2


# ─────────────────────────────────────────────
# 3. Frame Generator
# ─────────────────────────────────────────────
def gen_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️  Failed to grab frame.")
            break

        annotated_frame = frame.copy()

        if model is not None:
            try:
                # YOLO26 is NMS-free — no need to set iou or agnostic_nms.
                # End-to-end inference handles post-processing internally.
                results = model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False
                )

                for result in results:
                    # ── Detection / Segmentation ──────────────────────
                    if result.boxes is not None and len(result.boxes):
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf  = float(box.conf[0])
                            cls   = int(box.cls[0])
                            label = model.names.get(cls, str(cls))

                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                                          BOX_COLOR, BOX_THICKNESS)

                            display_text = f"{label} {conf:.2f}"
                            (tw, th), _ = cv2.getTextSize(
                                display_text, FONT, FONT_SCALE, FONT_THICKNESS)

                            # Background rectangle for text readability
                            cv2.rectangle(annotated_frame,
                                          (x1, y1 - th - 10), (x1 + tw + 6, y1),
                                          BOX_COLOR, -1)
                            cv2.putText(annotated_frame, display_text,
                                        (x1 + 3, y1 - 5), FONT, FONT_SCALE,
                                        (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)

                    # ── Classification (no boxes) ─────────────────────
                    elif result.probs is not None:
                        top_cls   = int(result.probs.top1)
                        top_conf  = float(result.probs.top1conf)
                        label     = model.names.get(top_cls, str(top_cls))
                        display_text = f"{label}: {top_conf:.2f}"

                        cv2.putText(annotated_frame, display_text,
                                    (20, 50), FONT, 1.2,
                                    TEXT_COLOR, 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Inference error: {e}")

        # Encode to JPEG and stream
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


# ─────────────────────────────────────────────
# 4. Flask Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ─────────────────────────────────────────────
# 5. Entry Point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)