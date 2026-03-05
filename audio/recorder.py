import logging
import numpy as np
import sounddevice as sd
from config import AudioConfig

logger = logging.getLogger(__name__)

TARGET_SR = 16000  # Required by openwakeword and faster-whisper

def _resample(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Linear interpolation resample to TARGET_SR. No extra dependencies."""
    if orig_sr == TARGET_SR:
        return audio
    target_len = int(len(audio) * TARGET_SR / orig_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, target_len),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)

class AudioRecorder:
    def __init__(self, config: AudioConfig):
        self.config = config
        if self.config.input_device is None:
            logger.info("Audio input  → auto-detect")
        else:
            logger.info("Audio input  → device index %d", self.config.input_device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream_chunks(self):
        """
        Infinite generator yielding float32 mono arrays resampled to
        TARGET_SR (16000 Hz), each representing ~80 ms of audio.
        """
        device   = self.config.input_device
        channels = self.config.channels

        logger.info("Opening wake-word stream  device=%s", device)

        with sd.InputStream(
            samplerate=None,
            channels=channels,
            dtype="float32",
            blocksize=native_chunk,
            device=device,
        ) as stream:
            actual_sr    = int(stream.samplerate)
            actual_chunk = int(self.config.chunk_size * actual_sr / TARGET_SR)
            logger.info(
                "Stream opened  actual_sr=%d  chunk=%d → resample to %d Hz  chunk=%d",
                actual_sr, actual_chunk, TARGET_SR, self.config.chunk_size,
            )
            while True:
                chunk, _ = stream.read(actual_chunk)
                yield _resample(chunk.flatten(), actual_sr)

    def record_until_silence(
        self,
        silence_duration: float,
        max_seconds: float,
        silence_threshold: float,
    ) -> np.ndarray:
        """
        Records a single utterance and returns a float32 mono array
        resampled to TARGET_SR (16000 Hz).
        """
        device   = self.config.input_device
        channels = self.config.channels

        logger.info(
            "Recording utterance  max=%.1fs  silence=%.1fs  threshold=%.4f",
            max_seconds, silence_duration, silence_threshold,
        )

        with sd.InputStream(
            samplerate=None,
            channels=channels,
            dtype="float32",
            device=device,
        ) as stream:
            actual_sr      = int(stream.samplerate)
            actual_chunk   = int(self.config.chunk_size * actual_sr / TARGET_SR)
            max_chunks     = int(max_seconds      * actual_sr / actual_chunk)
            silence_chunks = int(silence_duration * actual_sr / actual_chunk)

            recorded:     list[np.ndarray] = []
            silent_count: int              = 0

            for _ in range(max_chunks):
                chunk, _ = stream.read(actual_chunk)
                mono      = chunk.flatten()
                recorded.append(mono)

                rms = float(np.sqrt(np.mean(mono ** 2)))

                if rms < silence_threshold:
                    silent_count += 1
                    if silent_count >= silence_chunks and len(recorded) > silence_chunks * 2:
                        logger.debug("Silence detected — stopping recording")
                        break
                else:
                    silent_count = 0

        raw   = np.concatenate(recorded)
        audio = _resample(raw, actual_sr)
        logger.info(
            "Captured %.2f s  resampled %d Hz → %d Hz",
            len(audio) / TARGET_SR, actual_sr, TARGET_SR,
        )
        return audio
