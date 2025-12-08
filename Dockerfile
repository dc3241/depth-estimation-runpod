FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

RUN python -c "import torch; torch.hub.load('LiheYoung/Depth-Anything', 'Depth_Anything_V2_Small', pretrained=True)"

CMD ["python", "-u", "handler.py"]
