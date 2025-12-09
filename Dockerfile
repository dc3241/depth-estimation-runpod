FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

RUN apt-get update && apt-get install -y wget git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# Clone Depth-Anything repository
RUN git clone https://github.com/LiheYoung/Depth-Anything /app/Depth-Anything

# Create __init__.py file for the depth_anything module
RUN touch /app/Depth-Anything/__init__.py && \
    touch /app/Depth-Anything/depth_anything/__init__.py

# Verify setup
RUN ls -la /app/Depth-Anything/ && \
    echo "✓ Setup complete"

CMD ["python", "-u", "handler.py"]
