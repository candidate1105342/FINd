import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
import numpy as np
import io

from find.FINd_opt import FINDHasher

# set up logging to include timestamps and log levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# accepted file types for the /compare endpoint
allowed_content_types = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp", "image/tiff"}
max_file_size = 10 * 1024 * 1024  # 10 MB


# documents the response schema in /docs and validates the returned hamming_distance is within the expected 0-256 range
class CompareResponse(BaseModel):
    hamming_distance: int = Field(..., ge=0, le=256, description="Hamming distance between the two image hashes")


# reject uploads that are not images before reading the body
def validate_image_upload(upload: UploadFile) -> None:
    if not upload.content_type or upload.content_type not in allowed_content_types:
        logger.warning("Rejected '%s': unsupported or missing file type '%s'", upload.filename, upload.content_type)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported or missing file type. Must be one of: {', '.join(sorted(allowed_content_types))}",
        )

# check declared size before reading to avoid loading oversized files into memory
async def read_with_limit(upload: UploadFile, limit: int) -> bytes:
    if upload.size is not None and upload.size > limit:
        logger.warning("Rejected '%s': declared size %d bytes exceeds limit", upload.filename, upload.size)
        raise HTTPException(413, f"File '{upload.filename}' exceeds the {limit // (1024 * 1024)} MB size limit")
    data = await upload.read()
    # recheck after reading in case declared size was misreported
    if len(data) > limit:
        logger.warning("Rejected '%s': actual size %d bytes exceeds limit", upload.filename, len(data))
        raise HTTPException(413, f"File '{upload.filename}' exceeds the {limit // (1024 * 1024)} MB size limit")
    return data


hasher = FINDHasher()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FINd hasher initialised and ready")
    yield


app = FastAPI(lifespan=lifespan)


# compute a FINd perceptual hash and return it as a flat uint8 array
def compute_hash(img):
    h = hasher.fromImage(img)
    return np.array(h.hash, dtype=np.uint8).ravel()

# count differing bits between two hash arrays
def hamming(a, b):
    return int(np.sum(a ^ b))

@app.post("/compare", response_model=CompareResponse)
async def compare(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    logger.info("Received compare request: %s vs %s", image1.filename, image2.filename)

    # validate file type before reading, then read with size enforcement
    validate_image_upload(image1)
    validate_image_upload(image2)
    data1 = await read_with_limit(image1, max_file_size)
    data2 = await read_with_limit(image2, max_file_size)

    try:
        # decode files into PIL image objects
        img1 = Image.open(io.BytesIO(data1))
        img2 = Image.open(io.BytesIO(data2))
        # compute hashes and hamming distance
        h1 = compute_hash(img1)
        h2 = compute_hash(img2)
        distance = hamming(h1, h2)
    # corrupt or unrecognisable image data is a client error, not a server fault
    except UnidentifiedImageError:
        logger.warning("Invalid image data: %s vs %s", image1.filename, image2.filename)
        raise HTTPException(400, "One or more files could not be read as a valid image")
    # log errors with traceback for debugging and return a generic error message to the client
    except Exception as e:
        logger.error("Error processing request: %s", e, exc_info=True)
        raise HTTPException(500, "Failed to process images")

    logger.debug("FINd hashes: %s, %s", h1.tobytes().hex(), h2.tobytes().hex())
    logger.info("Hamming distance: %d", distance)
    return CompareResponse(hamming_distance=distance)
