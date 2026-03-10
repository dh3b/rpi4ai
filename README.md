# AI Assistant - Raspberry Pi 4B

A fully local voice assistant running on a Raspberry Pi 4B (8 GB).
No cloud services required after initial model downloads.

```
Wake Word  ->  STT (Whisper)  ->  LLM (llama.cpp)  ->  TTS (Piper)
```

---

## Project Structure

```
ai-assistant/
├── main.py                  # Pipeline assembler
├── config/__init__.py       # Env-based configuration
├── audio/
│   ├── recorder.py          # USB mic capture
│   └── speaker.py           # 3.5mm playback
├── wake_word/
│   └── detector.py          # openwakeword
├── stt/
│   └── transcriber.py       # faster-whisper
├── llm/
│   └── inference.py         # llama-cpp-python (GGUF)
├── tts/
│   └── synthesizer.py       # Piper TTS
├── models/
│   ├── llm/                 # Place .gguf file here
│   ├── wakeword/            # Place .onnx wake-word model here
│   └── tts/                 # Place .onnx + .onnx.json voice here
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Install Docker on Trixie

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in
```

### 2. Download models

**LLM — Llama 3.2 3B (recommended for 8 GB RPI):**
```bash
pip install huggingface_hub
huggingface-cli download \
  bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --local-dir models/llm/

mv models/llm/Llama-3.2-3B-Instruct-Q4_K_M.gguf models/llm/model.gguf
```

**TTS voice — Piper en_US-lessac-medium:**
```bash
wget -P models/tts/ \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx

wget -P models/tts/ \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

**Wake word — train a custom model:**

Option A - Use a web trainer
Option B - Use a built-in openwakeword model for testing

**STT — Whisper**
```bash
pip install huggingface_hub
huggingface-cli download \
  Systran/faster-whisper-tiny \
  --local-dir models/whisper/tiny
```

You can replace `Systran/faster-whisper-tiny` with another compatible Whisper model if you prefer, as long as you keep the directory layout under `models/whisper/`.

### 3. Configure

```bash
cp .env.example .env
# Edit .env - at minimum set LLM_MODEL_PATH, PIPER_MODEL_PATH, PIPER_CONFIG_PATH
```

Update `.env` for your model paths:
```
PIPER_MODEL_PATH=/models/tts/en_US-lessac-medium.onnx
PIPER_CONFIG_PATH=/models/tts/en_US-lessac-medium.onnx.json
LLM_MODEL_PATH=/models/llm/model.gguf
WHISPER_MODEL_PATH=/models/whisper/tiny
```

### 4. Build & run

```bash
# First build is slow (~30-45 min on the Pi) because llama-cpp-python
# compiles from source. Run on a fast machine with buildx if possible.
docker compose up --build
```

**Cross-compile on a fast machine:**
```bash
docker buildx build \
  --platform linux/arm64 \
  -t ai-assistant:latest \
  --push \
  .

# Then on the Pi:
docker compose pull && docker compose up
```

---

## Audio Device Setup

Find your USB mic and speaker device indices:
```bash
docker run --rm --device /dev/snd \
  python:3.11-slim \
  bash -c "pip install sounddevice -q && python -c 'import sounddevice; print(sounddevice.query_devices())'"
```

Set the indices in `.env`:
```
AUDIO_INPUT_DEVICE=1   # USB mic index
AUDIO_OUTPUT_DEVICE=0  # 3.5mm output (usually 0 on RPi)
```

If you get ALSA errors, ensure your user is in the `audio` group:
```bash
sudo usermod -aG audio $USER
```

---

## Tuning

| Goal | Setting |
|------|---------|
| Reduce RAM usage | Lower `LLM_N_CTX` (e.g. 2048) |
| Faster transcription | Use a smaller Whisper model under `WHISPER_MODEL_PATH` (e.g. `tiny`) |
| Better transcription | Use a larger Whisper model under `WHISPER_MODEL_PATH` (e.g. `small`) |
| Less false wake-word triggers | Raise `WAKE_WORD_THRESHOLD` (e.g. 0.7) |
| Cut off sooner after speaking | Lower `SILENCE_DURATION` (e.g. 1.0) |
| Reduce background noise cutoff | Raise `SILENCE_THRESHOLD` (e.g. 0.04) |
| More CPU for LLM | Raise `LLM_N_THREADS` (max 4 on RPi 4) |