from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AgentAction:
    """
    One tool invocation requested by the LLM.

    JSON shape:
      { "tool": "<tool_name>", "args": { ... } }
    """

    tool: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResponse:
    """
    The ONLY allowed top-level shape for LLM outputs in agent mode.

    JSON shape:
      {
        "say": "User-facing text to speak",
        "actions": [ { "tool": "...", "args": { ... } }, ... ],
        "meta": { ... }   // optional
      }
    """

    say: str
    actions: List[AgentAction] = field(default_factory=list)
    meta: Optional[Dict[str, Any]] = None


AGENT_RESPONSE_JSON_CONTRACT = """\
You MUST respond with a single valid JSON object and nothing else.

Required top-level keys:
- "say": string
- "actions": array
Optional top-level keys:
- "meta": object

Where each element of "actions" MUST be:
{ "tool": string, "args": object }

Rules:
- Do not wrap the JSON in markdown fences.
- Do not include any extra keys at the top level.
- If no tools are needed, respond with: {"say": "<your answer>", "actions": []}
"""

