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
        self.config    = config
        self.sr        = config.sample_rate
        self.chunk     = config.chunk_size

        if config.input_device is None:
            logger.info("Audio input  -> auto-detect")
        else:
            logger.info("Audio input  -> device index %d", config.input_device)

        logger.info(
            "Opening input stream  sr=%d  chunk=%d  device=%s",
            self.sr, self.chunk, config.input_device,
        )
        self._stream = sd.InputStream(
            samplerate=config.sample_rate,
            channels=config.channels,
            dtype="float32",
            blocksize=config.chunk_size,
            device=config.input_device,
        )
        self._stream.start()
        logger.info("Input stream open")

    def stream_chunks(self):
        """
        Infinite generator yielding float32 mono arrays resampled to
        TARGET_SR (16000 Hz). Used by the wake-word detection loop.
        """
        while True:
            chunk, _ = self._stream.read(self.chunk)
            yield _resample(chunk.flatten(), self.sr)

    def record_until_silence(
        self,
        silence_duration: float,
        max_seconds: float,
        silence_threshold: float,
    ) -> np.ndarray:
        """
        Records a single utterance from the shared stream and returns a
        float32 mono array resampled to TARGET_SR (16000 Hz).
        """
        max_chunks     = int(max_seconds      * self.sr / self.chunk)
        silence_chunks = int(silence_duration * self.sr / self.chunk)

        recorded:     list[np.ndarray] = []
        silent_count: int              = 0

        logger.info(
            "Recording utterance  max=%.1fs  silence=%.1fs  threshold=%.4f",
            max_seconds, silence_duration, silence_threshold,
        )

        for _ in range(max_chunks):
            chunk, _ = self._stream.read(self.chunk)
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
        audio = _resample(raw, self.sr)
        logger.info(
            "Captured %.2f s of audio (resampled from %d to %d Hz)",
            len(audio) / TARGET_SR, self.sr, TARGET_SR,
        )
        return audio