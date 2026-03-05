import io
import logging
import numpy as np
import sounddevice as sd
import soundfile as sf

from config import AudioConfig

logger = logging.getLogger(__name__)


class AudioSpeaker:
    def __init__(self, config: AudioConfig):
        self.config = config
        self._log_device_info()

    def _log_device_info(self):
        if self.config.output_device is None:
            logger.info("Audio output -> auto-detect (3.5mm)")
        else:
            logger.info("Audio output -> device index %d", self.config.output_device)

    def play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """
        Play a float32 numpy array through the output device.
        Blocks until playback is complete.
        """
        logger.info(
            "Playing %.2f s  sr=%d  device=%s",
            len(audio) / sample_rate,
            sample_rate,
            self.config.output_device,
        )
        sd.play(audio, samplerate=sample_rate, device=self.config.output_device)
        sd.wait()
        logger.debug("Playback finished")

    def play_wav_bytes(self, wav_bytes: bytes) -> None:
        buf = io.BytesIO(wav_bytes)
        data, sr = sf.read(buf, dtype="float32")
        self.play_audio(data, sr)
