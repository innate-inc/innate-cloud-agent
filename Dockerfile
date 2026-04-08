FROM python:3.10-slim

WORKDIR /app

COPY apps/innate-cloud-agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/innate-cloud-agent/src/ src/
COPY apps/innate-cloud-agent/run_server.py .

ENV PORT=8080
EXPOSE 8080

CMD ["python", "run_server.py"]
