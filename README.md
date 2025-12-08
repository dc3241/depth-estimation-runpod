# Depth Estimation Serverless Endpoint

RunPod serverless endpoint for generating depth maps using Depth-Anything-V2.

## Usage

Send POST request to your RunPod endpoint:
```json
{
  "input": {
    "image_url": "https://your-s3-bucket.s3.amazonaws.com/image.jpg"
  }
}
```

## Response
```json
{
  "depth_map_base64": "iVBORw0KGgoAAAANS...",
  "width": 1920,
  "height": 1080,
  "processing_time_ms": 5430
}
```