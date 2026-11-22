FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 YOLO_CONFIG_DIR=/tmp/ultralytics VEHICLE_AI_HOME=/app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# CPU image. GPU deployment is a separate target after model benchmarking.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install .
COPY app.py ./
COPY configs ./configs
RUN useradd --create-home --uid 10001 vehicle \
    && mkdir -p /app/runtime /app/models /app/data \
    && chown -R vehicle:vehicle /app/runtime
USER vehicle
EXPOSE 8000
CMD ["python", "-m", "vehicle_ai", "--help"]
