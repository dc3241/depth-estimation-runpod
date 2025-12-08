FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

RUN apt-get update && apt-get install -y wget git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

RUN git clone https://github.com/LiheYoung/Depth-Anything /app/Depth-Anything

RUN cd /tmp && \
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth && \
    mv depth_anything_v2_vits.pth /app/

CMD ["python", "-u", "handler.py"]
