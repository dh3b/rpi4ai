from __future__ import annotations

import logging
from typing import Any, List

from llama_cpp_agent import FunctionCallingAgent, LlamaCppFunctionTool, MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppPythonProvider

from llm.inference import LLMInference
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentController:
    def __init__(
        self,
        llm: LLMInference,
        registry: ToolRegistry,
        *,
        speak_intermediate: bool = True,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.speak_intermediate = bool(speak_intermediate)

        self._tts = None
        self._speaker = None

        provider = LlamaCppPythonProvider(self.llm.model)

        function_tools: List[LlamaCppFunctionTool] = []
        for spec in self.registry.all():
            function_tools.append(
                LlamaCppFunctionTool(spec.func)
            )

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
            allow_parallel_function_calling=True,
            send_message_to_user_callback=_send_message_to_user_callback,
            messages_formatter_type=MessagesFormatterType.LLAMA_3,
        )

    def run_turn(
        self,
        user_prompt: str,
        *,
        tts=None,
        speaker=None,
    ) -> str:
        """
        Execute one user turn via llama-cpp-agent's FunctionCallingAgent.

        If `tts` and `speaker` are provided, intermediate agent messages
        are spoken via the agent callback; if `speak_intermediate` is False,
        we only speak the final response here.
        """
        original_prompt = (user_prompt or "").strip()
        if not original_prompt:
            return ""

        # Make TTS / speaker available to the callback for this turn.
        self._tts = tts
        self._speaker = speaker

        logger.info("Agent(run_turn) ← %s chars", len(original_prompt))
        response = self._agent.generate_response(original_prompt)

        if isinstance(response, str):
            final_say = response.strip()
        else:
            final_say = str(response).strip()

        if (not self.speak_intermediate) and final_say and tts is not None and speaker is not None:
            audio, sr = tts.synthesize(final_say)
            speaker.play_audio(audio, sr)

        return final_say

