import runpod
import torch
import numpy as np
from PIL import Image
import base64
import io
import urllib.request
import time
import sys
import os

print("="*60)
print("DEPTH-ANYTHING V2 HANDLER")
print("="*60)

# Add to path
sys.path.insert(0, '/app/Depth-Anything')

print("Importing Depth-Anything V2...")
# The correct import path based on the actual repo structure
from depth_anything_v2.dpt import DepthAnythingV2

print("✓ Import successful")

print("Loading model...")
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = DepthAnythingV2(**model_configs['vits'])
model.load_state_dict(torch.load('/app/depth_anything_v2_vits.pth', map_location='cpu'))
model = model.to(device).eval()
print("✓ Model ready")
print("="*60)

def download_image(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as response:
        image_data = response.read()
    return Image.open(io.BytesIO(image_data)).convert('RGB')

def handler(job):
    start_time = time.time()
    
    try:
        job_input = job.get('input', {})
        image_url = job_input.get('image_url')
        
        if not image_url:
            return {"error": "Missing image_url", "status": "failed"}
        
        print(f"Processing: {image_url}")
        
        # Download and convert
        download_start = time.time()
        image = download_image(image_url)
        download_time = (time.time() - download_start) * 1000
        print(f"Downloaded in {download_time:.0f}ms")
        
        image_np = np.array(image)
        
        # Generate depth
        inference_start = time.time()
        with torch.no_grad():
            depth = model.infer_image(image_np)
        inference_time = (time.time() - inference_start) * 1000
        print(f"Inference in {inference_time:.0f}ms")
        
        # Normalize to 0-255
        encoding_start = time.time()
        depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        
        # Encode as PNG
        depth_image = Image.fromarray(depth_normalized, mode='L')
        buffer = io.BytesIO()
        depth_image.save(buffer, format='PNG')
        depth_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        encoding_time = (time.time() - encoding_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"✓ Processed in {total_time:.0f}ms")
        
        return {
            "depth_map_base64": depth_base64,
            "width": image.width,
            "height": image.height,
            "processing_time_ms": int(total_time),
            "timings": {
                "download_ms": int(download_time),
                "inference_ms": int(inference_time),
                "encoding_ms": int(encoding_time)
            },
            "status": "success"
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}

print("Starting RunPod handler...")
runpod.serverless.start({"handler": handler})
