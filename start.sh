#!/bin/bash
# Script ini akan mendownload model jika belum tersedia

MODEL_PATH="/app/yolo_service/best.pt"
MODEL_DIR="/app/yolo_service"

# Buat direktori jika belum ada
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading YOLO model..."
    
    # Menggunakan wget untuk mengunduh raw file dari GitHub
    # PERHATIAN: URL GitHub yang Anda gunakan tidak akan bekerja untuk file binary
    # GitHub URL untuk "View Raw" atau "Download" diperlukan untuk file binary seperti .pt
    wget -O "$MODEL_PATH" "https://github.com/RizkyyArdiansyah/skinburn-backend/raw/master/app/yolo_service/best.pt"
    
    # Jika mengalami masalah dengan GitHub, alternatif menggunakan Google Drive:
    # apt-get update && apt-get install -y python3-pip && pip install gdown
    # FILEID="your-file-id-here" # Ganti dengan ID file Google Drive Anda
    # gdown --id $FILEID -O "$MODEL_PATH"
    
    # Cek apakah unduhan berhasil
    if [ ! -f "$MODEL_PATH" ] || [ ! -s "$MODEL_PATH" ]; then
        echo "Error: Failed to download model. File not found or empty."
        echo "Please upload the model manually or fix the download URL."
        exit 1
    fi
    
    echo "Model downloaded successfully!"
else
    echo "Model already exists, skipping download."
fi


# Jalankan aplikasi dengan port yang benar
exec uvicorn app.main:app --host 0.0.0.0 --port 8000