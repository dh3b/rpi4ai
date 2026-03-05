import logging
import numpy as np
import sounddevice as sd
from config import AudioConfig

logger = logging.getLogger(__name__)

class AudioRecorder:
    def __init__(self, config: AudioConfig):
        self.config = config
        self._log_device_info()

    def _log_device_info(self):
        if self.config.input_device is None:
            logger.info("Audio input  -> auto-detect")
        else:
            logger.info("Audio input  -> device index %d", self.config.input_device)

    def stream_chunks(self):
        """
        Yields (float32 mono) numpy arrays of
        `chunk_size` samples.  Used by the wake-word detection loop.
        """
        sr         = self.config.sample_rate
        chunk_size = self.config.chunk_size
        device     = self.config.input_device
        channels   = self.config.channels

        logger.info(
            "Opening input stream  sr=%d  chunk=%d  device=%s",
            sr, chunk_size, device
        )

        with sd.InputStream(
            samplerate=sr,
            channels=channels,
            dtype="float32",
            blocksize=chunk_size,
            device=device,
        ) as stream:
            while True:
                chunk, _ = stream.read(chunk_size)
                yield chunk.flatten()

    def record_until_silence(
        self,
        silence_duration: float,
        max_seconds: float,
        silence_threshold: float,
    ) -> np.ndarray:
        """
        Opens a fresh input stream and records until either:
          - silence_duration seconds of consecutive silence detected, OR
          - max_seconds total recording time reached.

        Returns a flat float32 mono numpy array at the configured sample rate.
        """
        sr         = self.config.sample_rate
        chunk_size = self.config.chunk_size
        device     = self.config.input_device
        channels   = self.config.channels

        max_chunks     = int(max_seconds    * sr / chunk_size)
        silence_chunks = int(silence_duration * sr / chunk_size)

        recorded:    list[np.ndarray] = []
        silent_count: int             = 0

        logger.info(
            "Recording utterance  max=%.1fs  silence=%.1fs  threshold=%.4f",
            max_seconds, silence_duration, silence_threshold,
        )

        with sd.InputStream(
            samplerate=sr,
            channels=channels,
            dtype="float32",
            blocksize=chunk_size,
            device=device,
        ) as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_size)
                mono = chunk.flatten()
                recorded.append(mono)

                rms = float(np.sqrt(np.mean(mono ** 2)))

                if rms < silence_threshold:
                    silent_count += 1
                    # Only cut if we have at least some speech before the silence
                    if silent_count >= silence_chunks and len(recorded) > silence_chunks * 2:
                        logger.debug("Silence detected — stopping recording")
                        break
                else:
                    silent_count = 0

        audio = np.concatenate(recorded)
        logger.info("Captured %.2f s of audio", len(audio) / sr)
        return audio
