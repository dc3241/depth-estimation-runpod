FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

RUN apt-get update && apt-get install -y wget git tree && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# Clone Depth-Anything repository
RUN git clone https://github.com/LiheYoung/Depth-Anything /app/Depth-Anything

# Check the actual structure
RUN echo "=== Repository structure ===" && \
    ls -la /app/Depth-Anything/ && \
    echo "=== Looking for depth_anything_v2 ===" && \
    find /app/Depth-Anything -name "*.py" -type f | head -20

# Create __init__.py files where needed
RUN touch /app/Depth-Anything/__init__.py && \
    if [ -d "/app/Depth-Anything/depth_anything_v2" ]; then \
        touch /app/Depth-Anything/depth_anything_v2/__init__.py; \
    else \
        echo "depth_anything_v2 directory not found, creating it..."; \
        mkdir -p /app/Depth-Anything/depth_anything_v2; \
        touch /app/Depth-Anything/depth_anything_v2/__init__.py; \
    fi

# Download model
RUN cd /tmp && \
    wget --no-check-certificate https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth && \
    mv depth_anything_v2_vits.pth /app/

# Verify everything
RUN ls -la /app/ && \
    ls -la /app/Depth-Anything/ && \
    echo "✓ Setup complete"

CMD ["python", "-u", "handler.py"]
