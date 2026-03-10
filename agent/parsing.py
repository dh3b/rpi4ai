from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.types import AgentAction, AgentResponse

logger = logging.getLogger(__name__)


def _try_load_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}

def parse_agent_response(raw_text: str) -> AgentResponse:
    """
    Parse the LLM output into an AgentResponse.

    If parsing fails, fall back to treating the entire output as plain speech and no actions.
    """
    text = (raw_text or "").strip()
    if not text:
        return AgentResponse(say="", actions=[])

    obj = _try_load_json(text)

    if not obj:
        logger.warning("Agent JSON parse failed; falling back to plain text")
        return AgentResponse(say=text, actions=[])

    say = str(obj.get("say", ""))

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

