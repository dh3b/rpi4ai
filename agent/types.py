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
You must respond with JSON that matches the schema implied by the tools.
The JSON object must follow this structure:

{
  "say": string,
  "actions": [
    { "tool": string, "args": object }
  ],
  "meta": object (optional)
}

Rules:
- Use "say" for user-facing text.
- Use "actions" to specify any tools you want to invoke, with their arguments.
- Result of tool calls will be passed back to you in the next prompt.

If no tools are required, return:
{"say": "<response text>", "actions": []}
If no response text is needed, return:
{"say": "", "actions": [ ... ]}
"""

