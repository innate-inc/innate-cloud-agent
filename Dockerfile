# Use a Python base image
FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual code
COPY src/ src/
COPY run_server.py .

# Cloud Run sets PORT=8080 by default
ENV PORT=8080
EXPOSE 8080

# Define the command to run the server
CMD ["python", "run_server.py"]
