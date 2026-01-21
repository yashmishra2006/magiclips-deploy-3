# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install system dependencies for video processing and OpenCV
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements and install dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/tmp_uploads

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV UPLOAD_DIR=/app/tmp_uploads

# Expose port
EXPOSE 8080

# Start command - run FastAPI with uvicorn
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]