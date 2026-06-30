# Cloud Run Deployment

Build, push, and deploy the Cloud Agent image to Google Cloud Run. These steps
target the internal `innate-agent` GCP project and its private Artifact
Registry.

## Build the Docker image

```bash
docker build \
  --platform=linux/amd64 \
  -t us-central1-docker.pkg.dev/innate-agent/innate-agent-websocket-server/agent-ws-server-image:v1.1.6 \
  .
```

## Push the Docker image to Google Artifact Registry

```bash
docker push us-central1-docker.pkg.dev/innate-agent/innate-agent-websocket-server/agent-ws-server-image:v1.1.6
```

## Deploy the Cloud Run service

```bash
gcloud run deploy agent-ws-server \
  --image us-central1-docker.pkg.dev/innate-agent/innate-agent-websocket-server/agent-ws-server-image:v1.1.6 \
  --platform managed \
  --region us-central1 \
  --port 8765
```

## Test the Cloud Run service

Use the `check_ws_server.py` script to test the Cloud Run service.

```bash
python3 check_ws_server.py
```
