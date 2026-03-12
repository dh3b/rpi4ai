from __future__ import annotations

import logging
from typing import Any, List, Optional

from llama_cpp_agent import FunctionCallingAgent, LlamaCppFunctionTool, MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppPythonProvider

from llm.inference import LLMInference
from tools.registry import ToolRegistry
from tts.synthesizer import TTSSynthesizer
from audio.speaker import AudioSpeaker

logger = logging.getLogger(__name__)


class AgentController:
    def __init__(
        self,
        llm: LLMInference,
        registry: ToolRegistry,
        *,
        speak_intermediate: bool = False,
        tts: Optional[TTSSynthesizer] = None,
        speaker: Optional[AudioSpeaker] = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.speak_intermediate = speak_intermediate

        self._tts = tts
        self._speaker = speaker

        provider = LlamaCppPythonProvider(self.llm.model)
        function_tools: List[LlamaCppFunctionTool] = [LlamaCppFunctionTool(spec.func) for spec in self.registry.all()]

        def _send_message_to_user_callback(message: str, **_kwargs: Any) -> None:
            """
            Callback invoked by llama-cpp-agent whenever there is a user-facing message.
            We optionally stream intermediate speech via TTS.
            """
            text = (message or "").strip()
            if not text:
                return

            logger.info("Agent(message) → %s", text)

            if self.speak_intermediate and self._tts is not None and self._speaker is not None:
                audio, sr = self._tts.synthesize(text)
                self._speaker.play_audio(audio, sr)

        self._agent = FunctionCallingAgent(
            provider,
            llama_cpp_function_tools=function_tools,
            system_prompt=self.llm.config.system_prompt,
            allow_parallel_function_calling=True,
            send_message_to_user_callback=_send_message_to_user_callback,
            messages_formatter_type=MessagesFormatterType.LLAMA_3,
        )

    def run_turn(
        self,
        user_prompt: str,
    ) -> str:
        """
        Execute one user turn via llama-cpp-agent's FunctionCallingAgent.
        """
        prompt = (user_prompt or "").strip()
        if not prompt:
            return ""

        logger.info("Agent(run_turn) ← %s chars", len(prompt))
        response = str(self._agent.generate_response(prompt).strip())

        return response

