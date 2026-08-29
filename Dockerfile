FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variables for memory efficiency
ENV PORT=10000
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TF_ENABLE_ONEDNN_OPTS=0

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 120 web_app:app"]
