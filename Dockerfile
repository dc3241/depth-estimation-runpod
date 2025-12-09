FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

RUN apt-get update && apt-get install -y wget git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# Clone Depth-Anything repository
RUN git clone https://github.com/LiheYoung/Depth-Anything /app/Depth-Anything

# FIX: Create __init__.py files using a simple echo command
RUN echo "" > /app/Depth-Anything/__init__.py && \
    echo "" > /app/Depth-Anything/depth_anything_v2/__init__.py && \
    ls -la /app/Depth-Anything/__init__.py && \
    ls -la /app/Depth-Anything/depth_anything_v2/__init__.py

# Download model
RUN cd /tmp && \
    wget --no-check-certificate https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth && \
    mv depth_anything_v2_vits.pth /app/

# Verify everything
RUN ls -la /app/ && \
    ls -la /app/Depth-Anything/ && \
    ls -la /app/Depth-Anything/depth_anything_v2/ && \
    echo "✓ Setup complete"

CMD ["python", "-u", "handler.py"]
