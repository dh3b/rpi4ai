FROM python:3.11-slim-bookworm

# dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    alsa-utils \
    wget \
    tar \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Piper TTS
ARG PIPER_VERSION=2023.11.14-2

RUN set -eux; \
    ARCH=$(dpkg --print-architecture); \
    # Map debian arch names -> piper release names
    case "$ARCH" in \
      arm64)   PIPER_ARCH="aarch64" ;; \
      armhf)   PIPER_ARCH="armv7l"  ;; \
      amd64)   PIPER_ARCH="x86_64"  ;; \
      *)       echo "Unsupported arch: $ARCH" && exit 1 ;; \
    esac; \
    wget -qO /tmp/piper.tar.gz \
      "https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_${PIPER_ARCH}.tar.gz"; \
    mkdir -p /tmp/piper_extract; \
    tar -xzf /tmp/piper.tar.gz -C /tmp/piper_extract; \
    # The tarball extracts to piper/piper
    cp /tmp/piper_extract/piper/piper /usr/local/bin/piper; \
    chmod +x /usr/local/bin/piper; \
    rm -rf /tmp/piper.tar.gz /tmp/piper_extract

# python dependencies
WORKDIR /app

COPY requirements.txt .

# llama-cpp-python compiles llama.cpp from source.
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
ENV FORCE_CMAKE=1

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -c "import openwakeword; openwakeword.utils.download_models()"

# app code
COPY . .

# non-root user with audio group access 
RUN useradd -m -s /bin/bash assistant && \
    usermod -aG audio assistant && \
    chown -R assistant:assistant /app

USER assistant

# model cache dirs (Whisper auto-downloads here)
ENV HF_HOME=/app/.cache/huggingface
ENV XDG_CACHE_HOME=/app/.cache

# pre-download whisper
ARG WHISPER_MODEL_SIZE=base
RUN python -c "\
from faster_whisper import WhisperModel; \
import os; \
print('Downloading Whisper model:', os.environ['WHISPER_MODEL_SIZE']); \
WhisperModel('${WHISPER_MODEL_SIZE}', device='cpu', compute_type='int8'); \
print('Whisper model cached.')"

CMD ["python", "main.py"]
