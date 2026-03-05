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
        self._stream: sd.InputStream | None = None

        if self.config.input_device is None:
            logger.info("Audio input  -> auto-detect")
        else:
            logger.info("Audio input  -> device index %d", self.config.input_device)

    def _get_stream(self) -> sd.InputStream:
        """
        Returns the open input stream, creating it if necessary.
        A single stream is shared between stream_chunks() and
        record_until_silence() so ALSA never sees two simultaneous opens
        of the same device (which causes -9985 Device unavailable).
        """
        if self._stream is None or self._stream.closed:
            sr         = self.config.sample_rate
            chunk_size = self.config.chunk_size
            device     = self.config.input_device
            channels   = self.config.channels

            logger.info(
                "Opening input stream  sr=%d  chunk=%d  device=%s",
                sr, chunk_size, device
            )
            self._stream = sd.InputStream(
                samplerate=sr,
                channels=channels,
                dtype="float32",
                blocksize=chunk_size,
                device=device,
            )
            self._stream.start()
        return self._stream

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream_chunks(self):
        """
        Infinite generator yielding float32 mono arrays resampled to
        TARGET_SR (16000 Hz). Used by the wake-word detection loop.
        """
        sr         = self.config.sample_rate
        chunk_size = self.config.chunk_size
        stream     = self._get_stream()

        while True:
            chunk, _ = stream.read(chunk_size)
            yield _resample(chunk.flatten(), sr)

    def record_until_silence(
        self,
        silence_duration: float,
        max_seconds: float,
        silence_threshold: float,
    ) -> np.ndarray:
        """
        Records a single utterance on the shared stream and returns a
        float32 mono array resampled to TARGET_SR (16000 Hz).
        """
        sr         = self.config.sample_rate
        chunk_size = self.config.chunk_size
        stream     = self._get_stream()

        max_chunks     = int(max_seconds      * sr / chunk_size)
        silence_chunks = int(silence_duration * sr / chunk_size)

        recorded:     list[np.ndarray] = []
        silent_count: int              = 0

        logger.info(
            "Recording utterance  max=%.1fs  silence=%.1fs  threshold=%.4f",
            max_seconds, silence_duration, silence_threshold,
        )

        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
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
        audio = _resample(raw, sr)
        logger.info(
            "Captured %.2f s of audio (resampled from %d to %d Hz)",
            len(audio) / TARGET_SR, sr, TARGET_SR,
        )
        return audio