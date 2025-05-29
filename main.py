from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import cv2
import base64
import os
import requests

# FastAPI init
app = FastAPI()

origins = [
    "https://skinburn-detect.vercel.app",
    "https://www.skinburn-detect.vercel.app",
    "https://skinburn-detect-git-main.vercel.app",
    "https://skinburn-detect-*.vercel.app",  # Wildcard for preview deployments
    "http://localhost:5173",
    "http://localhost:3000",
]
# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download model if not exists
MODEL_PATH = "best.pt"
MODEL_URL = "https://github.com/RizkyyArdiansyah/skinburn-backend/blob/master/best.pt"  # Ganti ini dengan URL model kamu

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading YOLOv8 model...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
        print("Model downloaded.")

# Load model
download_model()
model = YOLO(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")

    # Convert to PIL image
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)[:, :, ::-1]  # RGB to BGR

    # Predict
    results = model.predict(img_np, conf=0.4, retina_masks=True, verbose=False)
    result = results[0]

    detections = []
    img_result = img_np.copy()

    for box in result.boxes:
        cls_id = int(box.cls.item())
        label = result.names[cls_id]
        confidence = round(float(box.conf.item()) * 100, 2)
        bbox = [round(float(x), 2) for x in box.xyxy[0].tolist()]

        # Draw box
        color = (0, 255, 0)
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_result, f"{label} {confidence:.1f}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        detections.append({
            "label": label,
            "confidence": confidence,
            "bbox": bbox
        })

    # Encode result image to base64
    _, buffer = cv2.imencode(".jpg", img_result)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "detections": detections,
        "result_image_base64": img_base64
    })

@app.get("/")
async def root():
    return {"message": "API for burn detection is running"}
