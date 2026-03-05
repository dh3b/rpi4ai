import logging
import numpy as np
from openwakeword.model import Model
from config import WakeWordConfig

logger = logging.getLogger(__name__)

class WakeWordDetector:
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self.model  = self._load_model()

    def _load_model(self) -> Model:
        logger.info(
            "Loading wake-word model  path=%s  framework=%s",
            self.config.model_path,
            self.config.inference_framework,
        )
        model = Model(
            wakeword_models=[self.config.model_path],
            inference_framework=self.config.inference_framework,
        )
        logger.info("Wake-word model ready")
        return model


    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Feed one float32 mono chunk into the model.
        Returns True (and resets internal state) when the wake phrase fires.

        audio_chunk : float32 numpy array, 16 kHz mono
        """

        pcm = np.clip(audio_chunk, -1.0, 1.0)
        pcm_int16 = (pcm * 32767).astype(np.int16)

        predictions: dict[str, float] = self.model.predict(pcm_int16)

        for model_name, score in predictions.items():
            if score >= self.config.threshold:
                logger.info(
                    "Wake word detected!  model=%s  score=%.3f  threshold=%.3f",
                    model_name, score, self.config.threshold,
                )
                self.reset()
                return True
        return False

    def reset(self) -> None:
        """Reset model state (call after false-positive or after each utterance)."""
        self.model.reset()
