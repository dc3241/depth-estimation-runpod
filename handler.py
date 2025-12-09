import runpod
import torch
import cv2
import numpy as np
from PIL import Image
import base64
import io
import urllib.request
import time
import sys
import os

# Add Depth-Anything to path
sys.path.insert(0, '/app/Depth-Anything')

print("Importing Depth-Anything V2...")
try:
    from depth_anything_v2.dpt import DepthAnythingV2
    print("✓ Import successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Available files in /app/Depth-Anything:")
    os.system("ls -la /app/Depth-Anything")
    raise

print("Loading model checkpoint...")
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

try:
    model = DepthAnythingV2(**model_configs['vits'])
    if os.path.exists('/app/depth_anything_v2_vits.pth'):
        model.load_state_dict(torch.load('/app/depth_anything_v2_vits.pth', map_location='cpu'))
        print("✓ Model loaded from /app/depth_anything_v2_vits.pth")
    else:
        print("✗ Model file not found at /app/depth_anything_v2_vits.pth")
        raise FileNotFoundError("Model checkpoint not found")
    
    model = model.cuda().eval()
    print("✓ Model moved to GPU and set to eval mode")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    raise

def download_image(url, max_retries=3):
    """Download image from URL with retry logic"""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                image_data = response.read()
            return Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise Exception(f"Failed to download image after {max_retries} attempts: {str(e)}")

def generate_depth_map(job):
    """RunPod handler function"""
    start_time = time.time()
    
    try:
        job_input = job.get('input', {})
        image_url = job_input.get('image_url')
        
        if not image_url:
            return {"error": "Missing 'image_url' in input", "status": "failed"}
        
        print(f"Processing: {image_url}")
        
        # Download
        download_start = time.time()
        image = download_image(image_url)
        download_time = (time.time() - download_start) * 1000
        print(f"Downloaded in {download_time:.0f}ms - Size: {image.size}")
        
        # Convert to numpy
        image_np = np.array(image)
        
        # Generate depth
        inference_start = time.time()
        with torch.no_grad():
            depth = model.infer_image(image_np)
        inference_time = (time.time() - inference_start) * 1000
        print(f"Inference: {inference_time:.0f}ms")
        
        # Normalize
        depth_min = depth.min()
        depth_max = depth.max()
        depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
        # Convert to image
        depth_image = Image.fromarray(depth_normalized, mode='L')
        
        # Encode
        encode_start = time.time()
        buffer = io.BytesIO()
        depth_image.save(buffer, format='PNG')
        depth_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        encode_time = (time.time() - encode_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"Total: {total_time:.0f}ms")
        
        return {
            "depth_map_base64": depth_base64,
            "width": image.width,
            "height": image.height,
            "processing_time_ms": int(total_time),
            "status": "success",
            "timings": {
                "download_ms": int(download_time),
                "inference_ms": int(inference_time),
                "encoding_ms": int(encode_time)
            }
        }
        
    except Exception as e:
        error_time = (time.time() - start_time) * 1000
        print(f"Error after {error_time:.0f}ms: {str(e)}")
        return {
            "error": str(e),
            "status": "failed",
            "processing_time_ms": int(error_time)
        }

print("Starting RunPod serverless handler...")
runpod
