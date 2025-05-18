#!/bin/bash
# Script ini akan mendownload model jika belum tersedia

MODEL_PATH="/app/yolo_service/best.pt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading YOLO model..."
    
    # UBAH INI dengan URL tempat Anda menyimpan model
    wget -O "$MODEL_PATH" "https://github.com/RizkyyArdiansyah/skinburn-backend/blob/master/app/yolo_service/best.pt"
    
    # Jika menggunakan Google Drive:
    # apt-get update && apt-get install -y python3-pip && pip install gdown
    # FILEID="your-file-id-here"
    # gdown --id $FILEID -O "$MODEL_PATH"
fi

PORT="${PORT:-8000}"
echo "Starting FastAPI application on port $PORT..."

# Jalankan aplikasi dengan port yang benar
exec uvicorn main:app --host 0.0.0.0 --port $PORT