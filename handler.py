import runpod
import torch
import cv2
import numpy as np
from PIL import Image
import base64
import io
import urllib.request
import time

print("Loading Depth-Anything-V2 model...")
model = torch.hub.load(
    'LiheYoung/Depth-Anything', 
    'Depth_Anything_V2_Small', 
    pretrained=True
)
model = model.cuda().eval()
print("Model loaded successfully!")

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
    """
    RunPod handler function for depth map generation
    
    Input format:
    {
        "input": {
            "image_url": "https://your-s3-bucket.s3.amazonaws.com/image.jpg"
        }
    }
    
    Output format:
    {
        "depth_map_base64": "iVBORw0KGgoAAAANS...",
        "width": 1920,
        "height": 1080,
        "processing_time_ms": 5430
    }
    """
    start_time = time.time()
    
    try:
        # Extract image URL from job input
        job_input = job.get('input', {})
        image_url = job_input.get('image_url')
        
        if not image_url:
            return {
                "error": "Missing 'image_url' in input",
                "status": "failed"
            }
        
        print(f"Processing image: {image_url}")
        
        # Download image
        download_start = time.time()
        image = download_image(image_url)
        download_time = (time.time() - download_start) * 1000
        print(f"Image downloaded in {download_time:.0f}ms - Size: {image.size}")
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Generate depth map
        inference_start = time.time()
        with torch.no_grad():
            depth = model.infer_image(image_np)
        inference_time = (time.time() - inference_start) * 1000
        print(f"Depth inference completed in {inference_time:.0f}ms")
        
        # Normalize depth map to 0-255
        depth_min = depth.min()
        depth_max = depth.max()
        depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
        # Convert to grayscale image
        depth_image = Image.fromarray(depth_normalized, mode='L')
        
        # Encode as base64
        encode_start = time.time()
        buffer = io.BytesIO()
        depth_image.save(buffer, format='PNG')
        depth_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        encode_time = (time.time() - encode_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"Total processing time: {total_time:.0f}ms")
        print(f"  - Download: {download_time:.0f}ms")
        print(f"  - Inference: {inference_time:.0f}ms")
        print(f"  - Encoding: {encode_time:.0f}ms")
        
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

# Start RunPod serverless handler
print("Starting RunPod serverless handler...")
runpod.serverless.start({"handler": generate_depth_map})
