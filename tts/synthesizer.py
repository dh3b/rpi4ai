import logging
import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from config import TTSConfig

logger = logging.getLogger(__name__)

class TTSSynthesizer:
    def __init__(self, config: TTSConfig):
        self.config = config
        self._verify_setup()

    def _verify_setup(self) -> None:
        # Verify piper binary is on PATH
        result = subprocess.run(
            ["which", "piper"], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise EnvironmentError(
                "piper binary not found on PATH. "
                "Ensure the Dockerfile installs it correctly."
            )

        # Verify model files
        for label, path in [
            ("model",  self.config.model_path),
            ("config", self.config.config_path),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Piper TTS {label} not found: {path}\n")

        logger.info("TTS  model=%s", self.config.model_path)

    def _build_piper_cmd(self, output_path: str) -> list[str]:
        cmd = [
            "piper",
            "--model",       self.config.model_path,
            "--config",      self.config.config_path,
            "--output_file", output_path,
        ]
        if self.config.speaker_id is not None:
            cmd += ["--speaker", str(self.config.speaker_id)]
        return cmd

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthesize text to speech.

        Returns
        audio: float32 numpy array (mono)
        sample_rate: int
        """
        logger.info("TTS ← '%s'", text)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name

        try:
            result = subprocess.run(
                self._build_piper_cmd(out_path),
                input=text.encode("utf-8"),
                capture_output=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Piper exited {result.returncode}: "
                    f"{result.stderr.decode(errors='replace')}"
                )

            audio, sample_rate = sf.read(out_path, dtype="float32")
            logger.info("TTS → %.2f s  sr=%d", len(audio) / sample_rate, sample_rate)
            return audio, sample_rate

        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass
