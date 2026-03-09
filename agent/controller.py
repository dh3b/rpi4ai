from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from agent.parsing import parse_agent_response
from agent.prompting import build_agent_system_appendix
from agent.types import AgentAction, AgentResponse
from llm.inference import LLMInference
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _tool_result(action: AgentAction, ok: bool, output: Any = None, error: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "tool": action.tool,
        "args": action.args,
        "ok": ok,
    }
    if ok:
        payload["output"] = output
    else:
        payload["error"] = error or "Unknown error"
    return payload


class AgentController:
    def __init__(
        self,
        llm: LLMInference,
        registry: ToolRegistry,
        *,
        max_iterations: int = 5,
        speak_intermediate: bool = True,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.max_iterations = max(1, int(max_iterations))
        self.speak_intermediate = bool(speak_intermediate)

    def _call_llm(self, user_message: str) -> str:
        appendix = build_agent_system_appendix(self.registry)
        logger.info("LLM(agent) ← %s chars", len(user_message))
        return self.llm.chat(user_message, extra_system_prompt=appendix)

    def _run_tools(self, actions: List[AgentAction]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for action in actions:
            logger.info("Tool ← %s  args=%s", action.tool, action.args)
            if not self.registry.has(action.tool):
                results.append(_tool_result(action, ok=False, error=f"Unknown tool: {action.tool}"))
                continue

            try:
                out = self.registry.call(action.tool, action.args)
                logger.info("Tool → %s  ok", action.tool)
                results.append(_tool_result(action, ok=True, output=out))
            except Exception as e:
                logger.exception("Tool execution failed: %s", action.tool)
                results.append(_tool_result(action, ok=False, error=str(e)))

        return results

    def _build_followup_message(
        self,
        *,
        original_user_prompt: str,
        previous_agent_response: AgentResponse,
        tool_results: List[Dict[str, Any]],
    ) -> str:
        """
        Feed back prompt + previous LLM response + tool outputs.
        """
        previous_json = json.dumps(asdict(previous_agent_response), ensure_ascii=False)
        results_json = json.dumps(tool_results, ensure_ascii=False)

        return "\n".join(
            [
                "You previously responded with JSON and requested tools. Here is the context:",
                "",
                f"Original user prompt:\n{original_user_prompt}",
                "",
                f"Previous assistant JSON:\n{previous_json}",
                "",
                f"Tool results (in order):\n{results_json}",
                "",
                "Now respond again with a single JSON object following the required schema.",
            ]
        ).strip()

    def run_turn(
        self,
        user_prompt: str,
        *,
        tts=None,
        speaker=None,
    ) -> str:
        """
        Execute one user turn using iterative tool-calling until no more actions.

        If tts and speaker are provided, this will speak the agent's `say` text
        at each iteration (or only the final one if speak_intermediate is False).

        Returns the final `say` text (may be empty).
        """
        original_prompt = user_prompt.strip()
        if not original_prompt:
            return ""

        say_accum: List[str] = []

        # First LLM call is directly the user's prompt.
        raw = self._call_llm(original_prompt)
        logger.debug("LLM(agent) raw → %s", raw)
        agent_resp = parse_agent_response(raw)
        logger.info("Agent parsed: say=%d chars  actions=%d", len(agent_resp.say), len(agent_resp.actions))

        if agent_resp.say:
            if self.speak_intermediate and tts is not None and speaker is not None:
                audio, sr = tts.synthesize(agent_resp.say)
                speaker.play_audio(audio, sr)
            say_accum.append(agent_resp.say)

        for i in range(self.max_iterations):
            if not agent_resp.actions:
                # No more tool requests.
                break

            tool_results = self._run_tools(agent_resp.actions)
            followup = self._build_followup_message(
                original_user_prompt=original_prompt,
                previous_agent_response=agent_resp,
                tool_results=tool_results,
            )

            raw = self._call_llm(followup)
            logger.debug("LLM(agent) raw → %s", raw)
            agent_resp = parse_agent_response(raw)
            logger.info("Agent parsed: say=%d chars  actions=%d", len(agent_resp.say), len(agent_resp.actions))

            if agent_resp.say:
                if self.speak_intermediate and tts is not None and speaker is not None:
                    audio, sr = tts.synthesize(agent_resp.say)
                    speaker.play_audio(audio, sr)
                say_accum.append(agent_resp.say)

        else:
            # Iteration limit hit (for-loop exhausted).
            limit_msg = "I reached the maximum number of tool iterations; stopping now."
            if self.speak_intermediate and tts is not None and speaker is not None:
                audio, sr = tts.synthesize(limit_msg)
                speaker.play_audio(audio, sr)
            say_accum.append(limit_msg)

        final_say = "\n".join([s for s in say_accum if s]).strip()

        if (not self.speak_intermediate) and final_say and tts is not None and speaker is not None:
            audio, sr = tts.synthesize(final_say)
            speaker.play_audio(audio, sr)

        return final_say

