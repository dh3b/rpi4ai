from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.types import AgentAction, AgentResponse

logger = logging.getLogger(__name__)


def _try_load_json(s: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return True, obj
        return False, None
    except Exception:
        return False, None


def _extract_first_object_block(s: str) -> Optional[str]:
    """
    Best-effort extraction of a JSON object from a messy string by taking the
    substring between the first '{' and last '}'.
    """
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return s[start : end + 1]


def parse_agent_response(raw_text: str) -> AgentResponse:
    """
    Parse the LLM output into an AgentResponse.

    If parsing fails, fall back to treating the entire output as plain speech and no actions.
    """
    text = (raw_text or "").strip()
    if not text:
        return AgentResponse(say="", actions=[])

    ok, obj = _try_load_json(text)
    if not ok:
        candidate = _extract_first_object_block(text)
        if candidate:
            ok, obj = _try_load_json(candidate)

    if not ok or not obj:
        logger.warning("Agent JSON parse failed; falling back to plain text")
        return AgentResponse(say=text, actions=[])

    say = obj.get("say", "")
    if not isinstance(say, str):
        say = str(say)

    actions_raw = obj.get("actions", [])
    actions: List[AgentAction] = []
    if isinstance(actions_raw, list):
        for a in actions_raw:
            if not isinstance(a, dict):
                continue
            tool = a.get("tool", "")
            args = a.get("args", {})
            if not isinstance(tool, str) or not tool.strip():
                continue
            if not isinstance(args, dict):
                args = {}
            actions.append(AgentAction(tool=tool.strip(), args=args))

    meta = obj.get("meta", None)
    if meta is not None and not isinstance(meta, dict):
        meta = {"value": meta}

    return AgentResponse(say=say.strip(), actions=actions, meta=meta)

