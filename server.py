from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel
import numpy as np
from PIL import Image
import torch
import base64
import io
import uvicorn
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model once at startup
print("Loading model...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")
else:
    print("Running on CPU")

try:
    model = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True)
    print("Model loaded successfully!")
    print(
        f"Model device: {next(model.parameters()).device if hasattr(model, 'parameters') else 'Unknown'}"
    )
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print(f"Traceback: {traceback.format_exc()}")
    raise


class ImageRequest(BaseModel):
    image: str  # base64 encoded image


class ImageResponse(BaseModel):
    panels: list


def base64_to_np_array(base64_string):
    """Convert base64 string to numpy array image"""
    try:
        logger.info("Starting base64 to numpy conversion")
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        logger.info(f"Decoded base64 data, size: {len(image_data)} bytes")

        # Open image from bytes
        image = Image.open(io.BytesIO(image_data)).convert("L").convert("RGB")
        logger.info(f"Opened PIL image, size: {image.size}")

        # Convert to numpy array
        image_array = np.array(image)
        logger.info(f"Converted to numpy array, shape: {image_array.shape}")
        return image_array
    except Exception as e:
        logger.error(f"Error in base64_to_np_array: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@app.post("/process-image/", response_model=ImageResponse)
async def process_image(request: ImageRequest):
    try:
        logger.info("Starting image processing request")

        # Convert base64 to numpy array
        logger.info("Converting base64 to numpy array")
        image_array = base64_to_np_array(request.image)
        logger.info("Base64 conversion completed successfully")

        # Process image with model
        logger.info("Starting model prediction")
        with torch.no_grad():
            logger.info("Calling model.predict_detections_and_associations")
            results = model.predict_detections_and_associations([image_array])
            logger.info(f"Model prediction completed, results type: {type(results)}")
            logger.info(f"Results length: {len(results) if results else 'None'}")

        # Extract panels from first (and only) result
        logger.info("Extracting panels from results")
        if not results or len(results) == 0:
            logger.error("No results returned from model")
            raise Exception("No results returned from model")

        result = results[0]
        logger.info(
            f"First result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}"
        )

        if "panels" not in result:
            logger.error(
                f"'panels' key not found in result. Available keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}"
            )
            raise Exception("'panels' key not found in model results")

        panels = result["panels"]
        logger.info(f"Extracted panels, type: {type(panels)}")

        # Convert numpy arrays to lists for JSON serialization
        logger.info("Converting panels for JSON serialization")
        if isinstance(panels, np.ndarray):
            logger.info("Converting numpy array to list")
            panels = panels.tolist()
        elif isinstance(panels, list):
            logger.info(f"Panels is already a list with {len(panels)} items")
            # Convert any numpy arrays within the list to regular lists
            panels = [
                panel.tolist() if isinstance(panel, np.ndarray) else panel
                for panel in panels
            ]

        logger.info(
            f"Final panels type: {type(panels)}, length: {len(panels) if hasattr(panels, '__len__') else 'N/A'}"
        )
        logger.info("Image processing completed successfully")

        return ImageResponse(panels=panels)

    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
