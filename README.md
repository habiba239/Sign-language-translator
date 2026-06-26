# Sign Language Translator 🤟

A bidirectional American Sign Language (ASL) translation system that converts hand gestures to text and text to sign language images in real-time.

## Features

- **Sign → Text**: Real-time hand gesture recognition using webcam
- **Text → Sign**: Convert written text into ASL alphabet images
- Spell & grammar correction support
- Interactive Streamlit UI with FastAPI backend

## Project Structure

```
sign-language-translator/
├── app/
│   ├── UI.py               # Streamlit frontend
│   ├── api.py              # FastAPI backend
│   ├── help_func.py        # Landmark feature extraction utilities
│   ├── text_to_sign.py     # Text-to-sign image pipeline
│   ├── extract_features.py # Dataset processing & landmark extraction
│   └── sign_to_text.py     # MLP model training
├── requirements.txt
└── README.md
```

## Model Performance

| Metric | Value |
|--------|-------|
| Train Accuracy | 99.86% |
| Test Accuracy | 99.29% |
| CV Mean Accuracy | 99.26% |

## Dataset

[ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data) — 87,000 images, 29 classes (A-Z + SPACE, DELETE, NOTHING)

## Pretrained Models

Download models from [Google Drive](#) and place in `models/` folder:
- `asl_mlp.joblib`
- `asl_scaler.joblib`
- `asl_label_encoder.joblib`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**1. Start the API:**
```bash
uvicorn app.api:app --reload
```

**2. Run the UI:**
```bash
streamlit run app/UI.py
```

## Team & Contributions

This project was built as a team of 2.

**My contribution:**
- Sign-to-Text pipeline (`extract_features.py`, `sign_to_text.py`)
- Hand landmark feature engineering using MediaPipe
- MLP classifier training & evaluation (99.29% test accuracy)
- FastAPI endpoint for real-time sign prediction

**Teammate's contribution:**
- Text-to-Sign pipeline (`text_to_sign.py`)
- Spell & grammar correction module
- Streamlit UI

## Tech Stack

- **MediaPipe** — Hand landmark detection
- **Scikit-learn** — MLP classifier
- **FastAPI** — Backend API
- **Streamlit** — Frontend UI
- **OpenCV** — Real-time video processing
## Demo

[![Sign Language Translator Demo]](https://youtu.be/oKTJ7SF2bgg)

## Pretrained Models

Download models from [HuggingFace](hab200/asl-sign-language-mlp) and place in `models/` folder:
