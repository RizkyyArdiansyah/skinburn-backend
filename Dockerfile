# Stage 1: Builder
FROM python:3.9-slim as builder

WORKDIR /app

# Install dependensi untuk opencv dan lainnya
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependensi Python dengan opsi CPU-only untuk PyTorch
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.0.0+cpu torchvision==0.15.0+cpu
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Install minimal runtime libraries untuk OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy kode aplikasi saja (tanpa best.pt)
COPY app/ /app/
COPY start.sh /app/
RUN chmod +x /app/start.sh

