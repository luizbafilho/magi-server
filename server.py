from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel
import numpy as np
from PIL import Image
import torch
import base64
import io
import uvicorn

app = FastAPI()

# Load model once at startup
print("Loading model...")
model = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True)
print("Model loaded successfully!")


class ImageRequest(BaseModel):
    image: str  # base64 encoded image


class ImageResponse(BaseModel):
    panels: list


def base64_to_np_array(base64_string):
    """Convert base64 string to numpy array image"""
    try:
        # Decode base64 string
        image_data = base64.b64decode(base64_string)

        # Open image from bytes
        image = Image.open(io.BytesIO(image_data)).convert("L").convert("RGB")

        # Convert to numpy array
        image_array = np.array(image)
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@app.post("/process-image/", response_model=ImageResponse)
async def process_image(request: ImageRequest):
    try:
        # Convert base64 to numpy array
        image_array = base64_to_np_array(request.image)

        # Process image with model
        with torch.no_grad():
            results = model.predict_detections_and_associations([image_array])

        # Extract panels from first (and only) result
        panels = results[0]["panels"]

        # Convert numpy arrays to lists for JSON serialization
        if isinstance(panels, np.ndarray):
            panels = panels.tolist()
        elif isinstance(panels, list):
            # Convert any numpy arrays within the list to regular lists
            panels = [
                panel.tolist() if isinstance(panel, np.ndarray) else panel
                for panel in panels
            ]

        return ImageResponse(panels=panels)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
