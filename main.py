import logging
import signal
import sys
import json

import numpy as np
from config import AppConfig
from audio.recorder   import AudioRecorder
from audio.speaker    import AudioSpeaker
from wake_word.detector import WakeWordDetector
from stt.transcriber  import SpeechTranscriber
from llm.inference    import LLMInference
from tts.synthesizer  import TTSSynthesizer
from agent.controller import AgentController
from tools import default_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("assistant")

class AIAssistantPipeline:
    """
    Assembles and runs the full voice-assistant pipeline.

    The pipeline is a single-threaded event loop:
      1. Stream mic chunks -> WakeWordDetector
      2. On wake word -> record user utterance
      3. Transcribe utterance -> text
      4. LLM inference -> response text
      5. TTS synthesis -> audio
      6. Play audio -> speaker
      7. Back to step 1
    """

    def __init__(self):
        logger.info("Initialising AI Assistant")
        cfg = AppConfig()

        self.recorder  = AudioRecorder(cfg.audio)
        self.speaker   = AudioSpeaker(cfg.audio)
        self.wake_word = WakeWordDetector(cfg.wake_word)
        self.stt       = SpeechTranscriber(cfg.stt)
        self.llm       = LLMInference(cfg.llm)
        self.tts       = TTSSynthesizer(cfg.tts)

        self._agent_enabled = cfg.agent_enabled
        self._agent = None
        if self._agent_enabled:
            registry = default_registry()
            self._agent = AgentController(
                self.llm,
                registry,
                speak_intermediate=cfg.agent_speak_intermediate,
            )

        # Pipeline-level knobs (from env via AppConfig)
        self._silence_duration   = cfg.silence_duration
        self._max_record_seconds = cfg.max_record_seconds
        self._silence_threshold  = cfg.silence_threshold

        self._confirmation_beep  = cfg.confirmation_beep
        self._beep_frequency     = cfg.beep_frequency
        self._beep_duration      = cfg.beep_duration

        self._running = False
        logger.info("─── Pipeline ready ───")

    def _play_confirmation_beep(self) -> None:
        """
        Synthesise a short sine-wave beep and play it immediately.
        Generated in-process — no file I/O, no external dependency.
        Frequency and duration are configurable via BEEP_FREQUENCY / BEEP_DURATION.
        """
        if not self._confirmation_beep:
            return

        sr      = 22050
        t       = np.linspace(0, self._beep_duration, int(sr * self._beep_duration), endpoint=False)
        beep    = np.sin(2 * np.pi * self._beep_frequency * t).astype(np.float32)
        fade    = np.linspace(1.0, 0.0, len(beep))
        beep   *= fade
        self.speaker.play_audio(beep, sr)

    def _install_signal_handlers(self):
        def _shutdown(sig, _frame):
            logger.info("Received signal %s - shutting down", sig)
            self._running = False
            sys.exit(0)

        signal.signal(signal.SIGINT,  _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

    def run(self):
        self._install_signal_handlers()
        self._running = True

        logger.info("Listening for wake word")

        for audio_chunk in self.recorder.stream_chunks():
            if not self._running:
                break

            # Step 1: wake word detection
            if not self.wake_word.process_chunk(audio_chunk):
                continue

            logger.info("Wake word - recording command")
            self._play_confirmation_beep()

            # Step 2: record utterance
            utterance_audio = self.recorder.record_until_silence(
                silence_duration=self._silence_duration,
                max_seconds=self._max_record_seconds,
                silence_threshold=self._silence_threshold,
            )

            # Step 3: speech -> text
            user_text = self.stt.transcribe(utterance_audio)

            if not user_text:
                logger.info("No speech detected - resuming wake-word loop")
                self.wake_word.reset()
                continue

            # Step 4: Agent/LLM inference
            if self._agent_enabled and self._agent is not None:
                raw_response = self._agent.run_turn(user_text, tts=self.tts, speaker=self.speaker)
            else:
                raw_response = self.llm.chat(user_text)
            try:
                response_text = json.loads(raw_response)["say"]
            except json.JSONDecodeError:
                logger.warning("LLM response was not valid JSON - treating as plain text")
                response_text = raw_response

            if not response_text:
                logger.info("Empty LLM response - resuming wake-word loop")
                continue

            # Step 5/6: text -> speech -> play response (only in non-agent mode).
            if not (self._agent_enabled and self._agent is not None):
                audio_out, sample_rate = self.tts.synthesize(response_text)
                self.speaker.play_audio(audio_out, sample_rate)

            logger.info("Listening for wake word")

if __name__ == "__main__":
    pipeline = AIAssistantPipeline()
    pipeline.run()
