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


AGENT_RESPONSE_JSON_CONTRACT = """
You must respond with exactly one valid JSON object. Output only JSON and nothing else.

The JSON object must follow this structure:

{
  "say": string,
  "actions": [
    { "tool": string, "args": object }
  ],
  "meta": object (optional)
}

Rules:
- "say" is required and must contain the text spoken to the user.
- "actions" is required and must be an array.
- Each element of "actions" must be:
  { "tool": string, "args": object }
- "meta" is optional and must be an object if present.
- Do not add any other top-level keys.
- Do not wrap the JSON in markdown or code blocks.

If no tools are required, return:

{"say": "<response text>", "actions": []}
"""

