import logging
import signal
import sys

from config import AppConfig
from audio.recorder   import AudioRecorder
from audio.speaker    import AudioSpeaker
from wake_word.detector import WakeWordDetector
from stt.transcriber  import SpeechTranscriber
from llm.inference    import LLMInference
from tts.synthesizer  import TTSSynthesizer

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

        # Pipeline-level knobs (from env via AppConfig)
        self._silence_duration   = cfg.silence_duration
        self._max_record_seconds = cfg.max_record_seconds
        self._silence_threshold  = cfg.silence_threshold

        self._running = False
        logger.info("Pipeline ready")

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

            # Step 4: LLM inference
            response_text = self.llm.chat(user_text)

            if not response_text:
                logger.info("Empty LLM response - resuming wake-word loop")
                continue

            # Step 5: text -> speech
            audio_out, sample_rate = self.tts.synthesize(response_text)

            # Step 6: play response
            self.speaker.play_audio(audio_out, sample_rate)

            logger.info("Listening for wake word")

if __name__ == "__main__":
    pipeline = AIAssistantPipeline()
    pipeline.run()
