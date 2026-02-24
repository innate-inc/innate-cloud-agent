FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies (opencv-python-headless needs no system libs)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual code
COPY src/ src/
COPY run_server.py .

# Cloud Run sets PORT=8080 by default
ENV PORT=8080
EXPOSE 8080

# Define the command to run the server
CMD ["python", "run_server.py"]
