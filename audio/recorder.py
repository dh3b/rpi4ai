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
        self.device_sr = self._query_native_sr()
        self._log_device_info()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_native_sr(self) -> int:
        """Query the device's native sample rate so we never request a rate it rejects."""
        try:
            info   = sd.query_devices(self.config.input_device, kind="input")
            native = int(info["default_samplerate"])
            logger.info(
                "Device native sample rate: %d Hz%s",
                native,
                "" if native == TARGET_SR else f" (will resample to {TARGET_SR} Hz)",
            )
            return native
        except Exception as exc:
            logger.warning(
                "Could not query device sample rate (%s) — "
                "falling back to configured sr=%d",
                exc, self.config.sample_rate,
            )
            return self.config.sample_rate

    def _log_device_info(self):
        if self.config.input_device is None:
            logger.info("Audio input  → auto-detect  native_sr=%d", self.device_sr)
        else:
            logger.info(
                "Audio input  → device index %d  native_sr=%d",
                self.config.input_device, self.device_sr,
            )

    def _native_chunk(self) -> int:
        """
        Returns the blocksize to request from the device such that after
        resampling each yielded chunk equals config.chunk_size samples at
        TARGET_SR (i.e. ~80 ms of audio).
        """
        return int(self.config.chunk_size * self.device_sr / TARGET_SR)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream_chunks(self):
        """
        Infinite generator yielding float32 mono arrays of exactly
        config.chunk_size samples at TARGET_SR (16000 Hz).
        Used by the wake-word detection loop.
        """
        native_chunk = self._native_chunk()
        device       = self.config.input_device
        channels     = self.config.channels

        logger.info(
            "Opening wake-word stream  native_sr=%d  native_chunk=%d  "
            "→ target_sr=%d  target_chunk=%d  device=%s",
            self.device_sr, native_chunk,
            TARGET_SR, self.config.chunk_size,
            device,
        )

        with sd.InputStream(
            samplerate=self.device_sr,
            channels=channels,
            dtype="float32",
            blocksize=native_chunk,
            device=device,
        ) as stream:
            while True:
                chunk, _ = stream.read(native_chunk)
                mono      = chunk.flatten()
                yield _resample(mono, self.device_sr)

    def record_until_silence(
        self,
        silence_duration: float,
        max_seconds: float,
        silence_threshold: float,
    ) -> np.ndarray:
        """
        Records a single utterance, resampled to TARGET_SR (16000 Hz).
        Returns a flat float32 mono numpy array.
        """
        native_chunk   = self._native_chunk()
        device         = self.config.input_device
        channels       = self.config.channels

        max_chunks     = int(max_seconds      * self.device_sr / native_chunk)
        silence_chunks = int(silence_duration * self.device_sr / native_chunk)

        recorded:     list[np.ndarray] = []
        silent_count: int              = 0

        logger.info(
            "Recording utterance  max=%.1fs  silence=%.1fs  threshold=%.4f",
            max_seconds, silence_duration, silence_threshold,
        )

        with sd.InputStream(
            samplerate=self.device_sr,
            channels=channels,
            dtype="float32",
            blocksize=native_chunk,
            device=device,
        ) as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(native_chunk)
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
        audio = _resample(raw, self.device_sr)
        logger.info(
            "Captured %.2f s → resampled from %d Hz to %d Hz",
            len(audio) / TARGET_SR, self.device_sr, TARGET_SR,
        )
        return audio
