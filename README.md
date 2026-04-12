# Sign Language Detector

Real-time sign recognition and sentence building using YOLO11l, OpenCV, and Flask.

## Overview

This project detects sign language gestures from a webcam feed, converts detections into words, and builds natural English sentences with a lightweight NLP pipeline.

Core capabilities:

- Real-time detection with YOLO11l (Ultralytics)
- Browser-based live stream via Flask
- Word buffering + debounce for phrase completion
- NLTK-assisted sentence generation for restaurant/food-service vocabulary
- Session transcript API and local logging to predictions.txt
- Health endpoint for deployment checks: /health

## Current Tech Stack

- Python 3
- Flask
- OpenCV
- Ultralytics YOLO
- NLTK

## Project Structure

```text
Sign Language Detector/
|-- app.py
|-- train_yolo11l.py
|-- best.pt
|-- yolo11l.pt
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- predictions.txt
|-- templates/
|   `-- index.html
|-- static/
|   |-- css/
|   `-- js/
|-- Notebook/
|   |-- Notebook.ipynb
|   `-- sign-language-real-time-detection.ipynb
|-- American-sign-language-2/
|   |-- data_fixed.yaml
|   |-- data.yaml
|   |-- images/
|   `-- labels/
`-- runs/
```

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.

```bash
git clone https://github.com/Niteesh3011/Sign-Language-Detection.git
cd "Sign Language Detector"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run The Web App

Start the Flask server:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

## API Endpoints

- GET /video_feed: MJPEG webcam stream with detections
- GET /get_sentence: current words, preview, finalized sentence, transcript, NLP status
- POST /clear_sentence: clears in-memory session transcript
- GET /health: deployment health check (app status + model/NLP state)

## Deployment Ready Pipeline

This repository now includes production-ready deployment artifacts:

- Dockerfile: production container build using gunicorn
- .dockerignore: excludes local/large artifacts from image context
- GitHub Actions CI: syntax checks on push/PR
- GitHub Actions CD: builds and pushes container image to GHCR on main branch

Pipeline files:

- .github/workflows/ci.yml
- .github/workflows/docker-publish.yml

### Container Deploy (Recommended)

Build locally:

```bash
docker build -t sign-language-detector:latest .
```

Run locally:

```bash
docker run --rm -p 5000:5000 \
	-e PORT=5000 \
	-e MODEL_PATH=/app/best.pt \
	sign-language-detector:latest
```

If your model is not baked into the image, mount it and point MODEL_PATH:

```bash
docker run --rm -p 5000:5000 \
	-e PORT=5000 \
	-e MODEL_PATH=/models/best.pt \
	-v /absolute/path/to/models:/models \
	sign-language-detector:latest
```

### GitHub Container Registry (GHCR)

On each push to main, docker-publish.yml builds and publishes:

- ghcr.io/<owner>/sign-language-detector:latest
- ghcr.io/<owner>/sign-language-detector:sha-<commit>

Use that image directly on Render, Railway, Fly.io, Azure Container Apps, ECS, or any Docker host.

## Train YOLO11l

Use the training script:

```bash
python train_yolo11l.py
```

What it does:

- Trains YOLO11l on American-sign-language-2 dataset
- Uses fallback training profiles for lower-memory environments
- Runs validation on test split
- Copies final best model to project root as best.pt

## Notebook Usage

Notebook inference relies on loading best.pt and test images correctly.

Recent fix applied:

- Added explicit YOLO import in inference cell
- Added path-safe model/test directory resolution so the notebook works whether the kernel starts from project root or Notebook folder

If needed, run this pattern in the notebook before prediction:

```python
from pathlib import Path
from ultralytics import YOLO

project_root = Path.cwd()
if not (project_root / "best.pt").exists():
	project_root = project_root.parent

best_model = YOLO(str(project_root / "best.pt"))
test_dir = project_root / "American-sign-language-2" / "images" / "test"
results = best_model.predict(source=str(test_dir), save=True, conf=0.35)
```

## GitHub Deployment Notes

The repository now includes a root .gitignore configured for this project.

It excludes:

- Virtual environments
- Python cache/build/test artifacts
- Large model files (*.pt, *.pth, etc.)
- Training outputs (runs/, mlflow/, logs/)
- Large dataset image/label folders
- Local prediction logs and editor/system files

Suggested first push:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Notes

- Keep best.pt in project root (or update model path candidates in app.py).
- predictions.txt is generated locally during runtime and is ignored by git.
- Webcam permission is required for live detection.
- For cloud deployment, set MODEL_PATH as an environment variable when model is stored outside the app folder.

## License

MIT
