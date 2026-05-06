FROM python:3.11-slim

WORKDIR /app

# Install API dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy both API and library
COPY api /app/api
COPY find /app/find

# Install the library
RUN pip install /app/find

# Run from API directory
WORKDIR /app/api

# Expose required port
EXPOSE 8945

# Run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8945"]