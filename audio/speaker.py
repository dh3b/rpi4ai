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
        self.volume = config.volume
        self._log_device_info()

    def _log_device_info(self):
        if self.config.output_device is None:
            logger.info("Audio output -> auto-detect (3.5mm)  volume=%.2f", self.volume)
        else:
            logger.info("Audio output -> device index %d  volume=%.2f", self.config.output_device, self.volume)

    def play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """
        Play a float32 numpy array through the output device.
        Volume is applied as a simple scalar multiplier (0.0 – 2.0).
        Blocks until playback is complete.
        """
        logger.info(
            "Playing %.2f s  sr=%d  volume=%.2f  device=%s",
            len(audio) / sample_rate,
            sample_rate,
            self.volume,
            self.config.output_device,
        )
        sd.play(audio * self.volume, samplerate=sample_rate, device=self.config.output_device)
        sd.wait()
        logger.debug("Playback finished")

    def play_wav_bytes(self, wav_bytes: bytes) -> None:
        buf = io.BytesIO(wav_bytes)
        data, sr = sf.read(buf, dtype="float32")
        self.play_audio(data, sr)
