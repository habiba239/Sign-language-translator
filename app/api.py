from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import uuid
from text_to_sign import generate_sign_image

import joblib
import numpy as np

# Load ML components 
model = joblib.load("asl_mlp.joblib")
scaler = joblib.load("asl_scaler.joblib")
encoder = joblib.load("asl_label_encoder.joblib")

# FastAPI app 
app = FastAPI(title="Sign Language Translator API")

# Request schema
class FeaturesRequest(BaseModel):
    features: list   # list of floats (landmarks, angles, distances, ...)


@app.get("/")
def root():
    return {"message": "Welcome to Sign Language Translator API 🚀"}

#sign_to_text
@app.post("/sign_to_text")
def predict(req: FeaturesRequest):
    """
    Endpoint to convert sign into text .
    """

    # Convert features to numpy array
    X = np.array(req.features).reshape(1, -1)
    # Scale
    X_scaled = scaler.transform(X)
    # Predict
    y_pred = model.predict(X_scaled)
    label = encoder.inverse_transform(y_pred)[0]

    return {"prediction": label}

#text_to_sign
@app.post("/text_to_sign_image/preview")
def text_to_sign_image_preview(text: str, enable_correction: bool = True):
    try:
        if enable_correction:
            img, file_name = generate_sign_image(text)
        else:
            # Skip correction: just clean text minimally
            from text_to_sign import text_to_sign_image
            img = text_to_sign_image(text.lower())
            file_name = "translation.png"

        if img is None:
            return {"error": "No valid letters found in dataset."}
        
        # Save temporary file
        unique_file = f"{uuid.uuid4().hex}_{file_name}"
        img.save(unique_file)
        
        return FileResponse(path=unique_file, media_type="image/png", filename=file_name)
    
    except Exception as e:
        return {"error": str(e)}