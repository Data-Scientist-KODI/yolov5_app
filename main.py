import io
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from ultralytics import YOLO

app = FastAPI(title="YOLO TFLite Face Cropper API")

# Load model globally on startup to avoid re-loading per request
MODEL_PATH = "best_float16.tflite"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

@app.post("/crop-best-face/")
async def crop_best_face(file: UploadFile = File(...)):
    # Validate input file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")

    # Read image contents into memory
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.")

    # Run inference directly on the decoded numpy image
    results = model(image)

    best_box = None
    max_confidence = 0.0

    if results:
        result = results[0]
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence > max_confidence:
                max_confidence = confidence
                best_box = box

    if best_box is None:
        raise HTTPException(status_code=444, detail="No face detected in the uploaded image.")

    # Get bounding box coordinates and crop image
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    
    # Clip coordinates to avoid out-of-bounds array slicing
    height, width, _ = image.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    
    cropped_face = image[y1:y2, x1:x2]

    # Encode cropped face to JPEG byte buffer
    is_success, buffer = cv2.imencode(".jpg", cropped_face)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode cropped image.")

    # Return cropped face with confidence in headers
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"X-Confidence-Score": f"{max_confidence:.4f}"}
    )