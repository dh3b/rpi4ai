"""
config/__init__.py
Loads all configuration from environment variables.
Each dataclass maps to one pipeline layer.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _optional_int(env_key: str, default: int = -1) -> Optional[int]:
    """Return None (auto-detect) for negative values, else the int."""
    try:
        v = int(os.getenv(env_key, str(default)))
        return None if v < 0 else v
    except (ValueError, TypeError):
        return None


@dataclass
class AudioConfig:
    sample_rate: int            = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    channels: int               = int(os.getenv("AUDIO_CHANNELS", "1"))
    chunk_size: int             = int(os.getenv("AUDIO_CHUNK_SIZE", "1280"))
    volume: float               = float(os.getenv("AUDIO_VOLUME", "1.0"))
    input_device: Optional[int]  = field(default=None)
    output_device: Optional[int] = field(default=None)

    def __post_init__(self):
        self.input_device  = _optional_int("AUDIO_INPUT_DEVICE",  -1)
        self.output_device = _optional_int("AUDIO_OUTPUT_DEVICE", -1)


@dataclass
class WakeWordConfig:
    model_path:          str   = os.getenv("WAKE_WORD_MODEL_PATH",  "/models/wakeword/model.onnx")
    threshold:           float = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))
    inference_framework: str   = os.getenv("WAKE_WORD_FRAMEWORK",   "onnx")


@dataclass
class STTConfig:
    model_path:   str = os.getenv("WHISPER_MODEL_PATH",    "/models/whisper")
    language:     str = os.getenv("WHISPER_LANGUAGE",      "en")
    device:       str = os.getenv("WHISPER_DEVICE",        "cpu")
    compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE",  "int8")


@dataclass
class LLMConfig:
    model_path:    str   = os.getenv("LLM_MODEL_PATH",    "/models/llm/model.gguf")
    system_prompt: str   = os.getenv("LLM_SYSTEM_PROMPT", "You are a helpful AI assistant. Keep answers concise.")
    max_tokens:    int   = int(os.getenv("LLM_MAX_TOKENS",   "512"))
    temperature:   float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    n_ctx:         int   = int(os.getenv("LLM_N_CTX",       "4096"))
    n_threads:     int   = int(os.getenv("LLM_N_THREADS",   "4"))
    top_p:         float = float(os.getenv("LLM_TOP_P",       "0.95"))


@dataclass
class TTSConfig:
    model_path:  str            = os.getenv("PIPER_MODEL_PATH",   "/models/tts/voice.onnx")
    config_path: str            = os.getenv("PIPER_CONFIG_PATH",  "/models/tts/voice.onnx.json")
    speaker_id:  Optional[int]  = _optional_int("PIPER_SPEAKER_ID", -1)


@dataclass
class AppConfig:
    audio:             AudioConfig    = field(default_factory=AudioConfig)
    wake_word:         WakeWordConfig = field(default_factory=WakeWordConfig)
    stt:               STTConfig      = field(default_factory=STTConfig)
    llm:               LLMConfig      = field(default_factory=LLMConfig)
    tts:               TTSConfig      = field(default_factory=TTSConfig)

    # Agent / tool-calling mode
    agent_enabled: bool = os.getenv("AGENT_ENABLED", "true").lower() == "true"
    agent_max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
    agent_speak_intermediate: bool = os.getenv("AGENT_SPEAK_INTERMEDIATE", "true").lower() == "true"

    # Recording / VAD
    silence_duration:   float = float(os.getenv("SILENCE_DURATION",   "1.5"))
    max_record_seconds: float = float(os.getenv("MAX_RECORD_SECONDS", "15.0"))
    silence_threshold:  float = float(os.getenv("SILENCE_THRESHOLD",  "0.02"))

    # Confirmation beep
    confirmation_beep:  bool  = os.getenv("CONFIRMATION_BEEP", "true").lower() == "true"
    beep_frequency:     float = float(os.getenv("BEEP_FREQUENCY", "880.0"))
    beep_duration:      float = float(os.getenv("BEEP_DURATION",  "0.15"))
