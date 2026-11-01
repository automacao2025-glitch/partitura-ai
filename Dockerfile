FROM python:3.10-slim

# Instala ffmpeg e libsndfile1 no Linux para suportar qualquer MP3, M4A, WAV e OGG
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala as dependências leves e o motor ONNX
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps basic-pitch==0.4.0

COPY . .

# Porta dinâmica exigida pelo Render
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"
