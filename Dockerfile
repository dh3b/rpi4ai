# Build
FROM python:3.11-slim-bookworm AS builder

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    wget \
    tar \
    ca-certificates

ARG PIPER_VERSION=2023.11.14-2

RUN set -ux; \
    ARCH=$(dpkg --print-architecture); \
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
    mkdir -p /piper/lib /piper/bin /piper/espeak-ng-data; \
    cp /tmp/piper_extract/piper/piper /piper/bin/piper; \
    cp /tmp/piper_extract/piper/*.so* /piper/lib/; \
    cp -r /tmp/piper_extract/piper/espeak-ng-data /piper/espeak-ng-data; \
    chmod +x /piper/bin/piper; \
    rm -rf /tmp/piper.tar.gz /tmp/piper_extract

WORKDIR /build
COPY requirements.txt .

ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
ENV FORCE_CMAKE=1

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip wheel --wheel-dir /wheels -r requirements.txt

# Runtime
FROM python:3.11-slim-bookworm

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    alsa-utils \
    ca-certificates

COPY --from=builder /piper/bin/piper /usr/local/bin/piper
COPY --from=builder /piper/lib/ /usr/local/lib/
COPY --from=builder /piper/espeak-ng-data /usr/share/espeak-ng-data

RUN ldconfig

COPY --from=builder /wheels /wheels

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-index --find-links=/wheels /wheels/*

RUN python -c "import openwakeword; openwakeword.utils.download_models()"

RUN useradd -m -s /bin/bash assistant && usermod -aG audio assistant

WORKDIR /app
COPY --chown=assistant:assistant . .

USER assistant

CMD ["python", "main.py"]