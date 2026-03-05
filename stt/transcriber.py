import logging
import numpy as np
from faster_whisper import WhisperModel
from config import STTConfig

logger = logging.getLogger(__name__)

class SpeechTranscriber:
    def __init__(self, config: STTConfig):
        self.config = config
        self.model  = self._load_model()

    def _load_model(self) -> WhisperModel:
        logger.info(
            "Loading Whisper  size=%s  device=%s  compute_type=%s",
            self.config.model_size,
            self.config.device,
            self.config.compute_type,
        )
        model = WhisperModel(
            self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        logger.info("Whisper ready")
        return model

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a float32 mono numpy array to text.
        Uses VAD filter to ignore non-speech segments, reducing
        hallucinations on silent recordings.
        Returns an empty string if no speech was detected.
        """

        logger.info("Transcribing %.2f s of audio...", len(audio) / 16000)

        segments, info = self.model.transcribe(
            audio,
            language=self.config.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        # Materialise the lazy generator
        text = " ".join(seg.text.strip() for seg in segments).strip()

        logger.info(
            "Transcription [lang=%s  prob=%.2f]: '%s'",
            info.language,
            info.language_probability,
            text or "<empty>",
        )
        return text
