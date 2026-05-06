# FINd Perceptual Hashing API

A FastAPI service that compares two images using the FINd perceptual hashing algorithm and returns their Hamming distance.

## Prerequisites

- **Docker**: for building and running the API
- **Python >= 3.11**: for installing the library and running the tests

## How FINd works

FINd produces a 256-bit perceptual hash from an image in four steps:

1. **Luminance conversion**: the image is resized to at most 512×512 pixels and converted to a grayscale luminance matrix using standard RGB coefficients.
2. **Box filter smoothing**: a box filter (local mean) is applied to the luminance matrix to remove high-frequency noise, making the hash robust to minor edits.
3. **Decimation**: the smoothed matrix is downsampled to 64×64 by sampling evenly-spaced pixels.
4. **DCT and thresholding**: a 2D Discrete Cosine Transform reduces the 64×64 matrix to a 16×16 representation capturing low-frequency structure. Each of the 256 values is thresholded against the median to produce one bit, giving the final 256-bit hash.

Two images are compared by computing the **Hamming distance** between their hashes: the number of bits that differ. A distance of 0 means identical hashes; lower values indicate greater visual similarity.

## Installing the FINd library

From the project root, install the package in editable mode:

```bash
pip install -e find/
```

This installs both the original (`find.FINd`) and optimized (`find.FINd_opt`) implementations along with their dependencies (`numpy`, `pillow`, `imagehash`).

## Running unit tests

From the project root:

```bash
python -m pytest find/tests/
```

Or using `unittest` directly:

```bash
python -m unittest discover -s find/tests
```

The test suite verifies that the optimized implementation produces bit-identical hashes to the reference implementation across a range of image sizes, aspect ratios, and color modes.

## Build and run

From the project directory, build the Docker image:

```bash
docker build -t find-api .
```

Then start a container, mapping port 8945 on your machine to port 8945 inside the container:

```bash
docker run -p 8945:8945 find-api
```

The API will be available at `http://localhost:8945` while the container is running.

To run the container in the background (detached mode):

```bash
docker run -d -p 8945:8945 find-api
```

### Viewing logs

```bash
docker logs <container-id>
```

To follow logs live as requests come in:

```bash
docker logs -f <container-id>
```

Find the container ID with `docker ps`.

### Stopping the container

```bash
docker stop <container-id>
```

## Usage

Send a POST request to `/compare` with two images as form-data. Each image must be at most 10 MB and one of: JPEG, PNG, GIF, WebP, BMP, TIFF.

```bash
curl -X POST http://localhost:8945/compare \
  -F "image1=@image1.jpg" \
  -F "image2=@image2.jpg"
```

### Response

```json
{"hamming_distance": 40}
```

A Hamming distance of `0` means the images are identical. Lower values indicate greater similarity.

### Error responses

| Status | Cause |
|--------|-------|
| `400`  | Unsupported or missing file type |
| `400`  | File could not be read as a valid image (e.g. corrupt data) |
| `413`  | File exceeds the 10 MB size limit |
| `500`  | Unexpected server error |

## Interactive docs

FastAPI provides auto-generated docs at [http://localhost:8945/docs](http://localhost:8945/docs).

To test the `/compare` endpoint directly from the docs page:

1. Open [http://localhost:8945/docs](http://localhost:8945/docs) in a browser
2. Click the `POST /compare` row to expand it
3. Click **Try it out** in the top right of the expanded section
4. Upload an image for both `image1` and `image2` using the file pickers
5. Click **Execute**

The response will appear below, showing the Hamming distance and the full request that was sent.

## Evaluations

The `evaluations/` directory contains Jupyter notebooks for benchmarking the algorithms:

- `accuracy.ipynb`: classification (ROC, precision, recall, F1) and retrieval (P@k, MAP) metrics
- `comp_performance.ipynb`: runtime, CPU time, memory usage, and scalability across image sizes
