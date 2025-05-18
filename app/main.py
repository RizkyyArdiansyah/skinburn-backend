from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import io
import os
import uuid
import cv2
import numpy as np

app = FastAPI()

# CORS configuration
# CORS configuration - FIXED
origins = [
    "https://skinburn-detect.vercel.app",
    # Optional: Add these for local development
    "http://localhost:5173",
]

# Apply CORS middleware with explicit configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Add these two parameters for better CORS handling
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Folder upload dan static
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLO model
model = YOLO("yolo_service/best.pt")  # Pastikan model ini mendukung segmentasi

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Baca file gambar dari upload
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Konversi PIL ke numpy BGR (OpenCV)
        img_cv = np.array(image)
        # Convert RGB to BGR for OpenCV
        img_cv = img_cv[:, :, ::-1].copy()

        # Predict menggunakan model YOLO dengan mode segmentasi
        results = model.predict(source=img_cv, conf=0.4, retina_masks=True, verbose=False)
        result = results[0]

        # Buat salinan gambar untuk menggambar hasil
        img_result = img_cv.copy()
        
        # Gambar segmentasi mask secara manual jika ada
        if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
            for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
                cls_id = int(box.cls.item())
                label = result.names[cls_id]
                confidence = float(box.conf.item())
                
                # Dapatkan warna berbeda untuk setiap kelas (BGR format)
                color = (0, 255, 0)  # Default hijau
                if cls_id == 0:  # Asumsikan 0 = first degree
                    color = (0, 255, 255)  # Kuning
                elif cls_id == 1:  # Asumsikan 1 = second degree
                    color = (0, 165, 255)  # Orange
                elif cls_id == 2:  # Asumsikan 2 = third degree
                    color = (0, 0, 255)    # Merah
                
                # Gambar bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
                
                # Tambahkan label dan confidence
                text = f"{label}: {confidence:.2f}"
                cv2.putText(img_result, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Gambar mask segmentasi jika tersedia
                if mask.data is not None:
                    # Konversi mask ke binary image
                    mask_image = mask.data.cpu().numpy()[0]
                    mask_binary = (mask_image * 255).astype(np.uint8)
                    
                    # Gambar outline contour dari mask untuk visualisasi yang lebih jelas
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_result, contours, -1, color, 2)
                    
                    # Buat overlay mask dengan warna semi-transparan
                    overlay = img_result.copy()
                    for c in contours:
                        cv2.fillPoly(overlay, [c], color)
                    
                    # Gabungkan overlay dengan gambar asli
                    cv2.addWeighted(overlay, 0.3, img_result, 0.7, 0, img_result)
        else:
            # Fallback ke bounding box saja jika tidak ada masks
            for box in result.boxes:
                cls_id = int(box.cls.item())
                label = result.names[cls_id]
                confidence = float(box.conf.item())
                
                # Warna berdasarkan kelas
                color = (0, 255, 0)  # Default hijau
                if cls_id == 0:
                    color = (0, 255, 255)
                elif cls_id == 1:
                    color = (0, 165, 255)
                elif cls_id == 2:
                    color = (0, 0, 255)
                
                # Gambar bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
                
                # Tambahkan label dan confidence
                text = f"{label}: {confidence:.2f}"
                cv2.putText(img_result, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Generate nama file unik
        filename = f"{uuid.uuid4()}.jpg"
        filepath_original = os.path.join(UPLOAD_DIR, filename)
        filepath_result = os.path.join(UPLOAD_DIR, f"result_{filename}")

        # Simpan gambar asli
        with open(filepath_original, "wb") as f:
            f.write(contents)

        # Simpan gambar hasil dengan segmentation mask
        cv2.imwrite(filepath_result, img_result)

        # Parsing hasil deteksi ke JSON
        detections = []
        
        if hasattr(result, 'masks') and result.masks is not None:
            # Proses mask segmentasi
            for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
                cls_id = int(box.cls.item())
                label = result.names[cls_id]
                confidence = round(float(box.conf.item()) * 100, 2)
                bbox = [round(float(x), 2) for x in box.xyxy[0].tolist()]
                
                # Extract contours from mask for JSON response
                # Mengkonversi mask ke format yang bisa disimpan di JSON
                if mask.data is not None:
                    mask_data = mask.data.cpu().numpy()[0]
                    # Mengambil area mask untuk statistik (opsional)
                    mask_area = mask_data.sum()
                    # Konversi mask untuk contour
                    mask_binary = (mask_data * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Konversi contour untuk JSON
                    contour_points = []
                    if len(contours) > 0:
                        # Mengambil contour terbesar
                        max_contour = max(contours, key=cv2.contourArea)
                        # Simplifikasi contour untuk mengurangi ukuran data
                        epsilon = 0.005 * cv2.arcLength(max_contour, True)
                        approx_contour = cv2.approxPolyDP(max_contour, epsilon, True)
                        contour_points = approx_contour.reshape(-1, 2).tolist()
                    
                    detections.append({
                        "label": label,
                        "confidence": confidence,
                        "bbox": bbox,
                        "mask_area": float(mask_area),
                        "contour": contour_points
                    })
        else:
            # Fallback ke bounding box jika tidak ada mask
            for box in result.boxes:
                cls_id = int(box.cls.item())
                label = result.names[cls_id]
                confidence = round(float(box.conf.item()) * 100, 2)
                bbox = [round(float(x), 2) for x in box.xyxy[0].tolist()]
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": bbox
                })

        return JSONResponse({
            "detections": detections,
            "original_image": f"/static/uploads/{filename}",
            "result_image": f"/static/uploads/result_{filename}"
        })

    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

@app.get("/")
async def root():
    return {"message": "API untuk deteksi dan segmentasi luka bakar aktif"}