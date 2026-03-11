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
    sample_rate: int            = int(os.getenv("AUDIO_SAMPLE_RATE"))
    channels: int               = int(os.getenv("AUDIO_CHANNELS"))
    chunk_size: int             = int(os.getenv("AUDIO_CHUNK_SIZE"))
    volume: float               = float(os.getenv("AUDIO_VOLUME"))
    input_device: Optional[int]  = field(default=None)
    output_device: Optional[int] = field(default=None)

    def __post_init__(self):
        self.input_device  = _optional_int("AUDIO_INPUT_DEVICE")
        self.output_device = _optional_int("AUDIO_OUTPUT_DEVICE")


@dataclass
class WakeWordConfig:
    model_path:          str   = os.getenv("WAKE_WORD_MODEL_PATH")
    threshold:           float = float(os.getenv("WAKE_WORD_THRESHOLD"))
    inference_framework: str   = os.getenv("WAKE_WORD_FRAMEWORK")


@dataclass
class STTConfig:
    model_path:   str = os.getenv("WHISPER_MODEL_PATH")
    language:     str = os.getenv("WHISPER_LANGUAGE")
    device:       str = os.getenv("WHISPER_DEVICE")
    compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE")


@dataclass
class LLMConfig:
    model_path:    str   = os.getenv("LLM_MODEL_PATH")
    system_prompt: str   = os.getenv("LLM_SYSTEM_PROMPT")
    max_tokens:    int   = int(os.getenv("LLM_MAX_TOKENS"))
    temperature:   float = float(os.getenv("LLM_TEMPERATURE"))
    n_ctx:         int   = int(os.getenv("LLM_N_CTX"))
    n_threads:     int   = int(os.getenv("LLM_N_THREADS"))
    top_p:         float = float(os.getenv("LLM_TOP_P"))
    agent_enabled: bool = os.getenv("AGENT_ENABLED").lower() == "true"


@dataclass
class TTSConfig:
    model_path:  str            = os.getenv("PIPER_MODEL_PATH")
    config_path: str            = os.getenv("PIPER_CONFIG_PATH")
    speaker_id:  Optional[int]  = _optional_int("PIPER_SPEAKER_ID")


@dataclass
class AppConfig:
    audio:             AudioConfig    = field(default_factory=AudioConfig)
    wake_word:         WakeWordConfig = field(default_factory=WakeWordConfig)
    stt:               STTConfig      = field(default_factory=STTConfig)
    llm:               LLMConfig      = field(default_factory=LLMConfig)
    tts:               TTSConfig      = field(default_factory=TTSConfig)

    # Agent / tool-calling mode
    agent_enabled: bool = os.getenv("AGENT_ENABLED").lower() == "true"
    agent_speak_intermediate: bool = os.getenv("AGENT_SPEAK_INTERMEDIATE").lower() == "true"

    # Recording / VAD
    silence_duration:   float = float(os.getenv("SILENCE_DURATION"))
    max_record_seconds: float = float(os.getenv("MAX_RECORD_SECONDS"))
    silence_threshold:  float = float(os.getenv("SILENCE_THRESHOLD"))

    # Confirmation beep
    confirmation_beep:  bool  = os.getenv("CONFIRMATION_BEEP").lower() == "true"
    beep_frequency:     float = float(os.getenv("BEEP_FREQUENCY"))
    beep_duration:      float = float(os.getenv("BEEP_DURATION"))
