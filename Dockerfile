FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

RUN apt-get update && apt-get install -y wget git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# Clone Depth-Anything repository
RUN git clone https://github.com/LiheYoung/Depth-Anything /app/Depth-Anything

# Create __init__.py files for proper Python module structure
RUN touch /app/Depth-Anything/__init__.py && \
    touch /app/Depth-Anything/depth_anything/__init__.py

# Download DINOv2 backbone (required by Depth-Anything)
RUN mkdir -p /app/torchhub && \
    cd /app/torchhub && \
    git clone https://github.com/facebookresearch/dinov2.git facebookresearch_dinov2_main

CMD ["python", "-u", "/app/handler.py"]
