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
import cv2

sys.path.insert(0, '/app/Depth-Anything')
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

print("=" * 60)
print("DEPTH-ANYTHING HANDLER")
print("=" * 60)

print("Importing Depth-Anything...")
time.sleep(0.5)
print("✓ Import successful")

print("Loading model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vits14').to(device).eval()

# Set up image transform pipeline
transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

print("✓ Model loaded successfully")

def download_image(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as response:
        image_data = response.read()
    return Image.open(io.BytesIO(image_data)).convert('RGB')

def handler(job):
    start_time = time.time()
    
    try:
        image_url = job.get('input', {}).get('image_url')
        if not image_url:
            return {"error": "Missing image_url", "status": "failed"}
        
        # Download image
        download_start = time.time()
        image = download_image(image_url)
        original_width, original_height = image.size
        download_time = (time.time() - download_start) * 1000
        
        # Convert to numpy array
        image_np = np.array(image) / 255.0
        
        # Apply transforms
        h, w = image_np.shape[:2]
        image_transformed = transform({'image': image_np})['image']
        image_tensor = torch.from_numpy(image_transformed).unsqueeze(0).to(device)
        
        # Run inference
        inference_start = time.time()
        with torch.no_grad():
            depth = model(image_tensor)
        inference_time = (time.time() - inference_start) * 1000
        
        # Process depth map
        encoding_start = time.time()
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(original_height, original_width),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        depth_np = depth.cpu().numpy()
        depth_normalized = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255).astype(np.uint8)
        
        # Convert to image
        depth_image = Image.fromarray(depth_normalized, mode='L')
        buffer = io.BytesIO()
        depth_image.save(buffer, format='PNG')
        depth_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        encoding_time = (time.time() - encoding_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "depth_map_base64": depth_base64,
            "width": original_width,
            "height": original_height,
            "processing_time_ms": int(total_time),
            "timings": {
                "download_ms": int(download_time),
                "inference_ms": int(inference_time),
                "encoding_ms": int(encoding_time)
            },
            "status": "success"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}

runpod.serverless.start({"handler": handler})
