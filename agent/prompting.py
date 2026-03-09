from __future__ import annotations

from tools.registry import ToolRegistry
from agent.types import AGENT_RESPONSE_JSON_CONTRACT


def build_tools_section(registry: ToolRegistry) -> str:
    lines: list[str] = []
    lines.append("Available tools (functions you may call):")
    for spec in registry.all():
        lines.append(f"- {spec.signature}")
        if spec.description:
            for dl in spec.description.splitlines():
                lines.append(f"  {dl}".rstrip())
    return "\n".join(lines).strip()


def build_agent_system_appendix(registry: ToolRegistry) -> str:
    """
    Text appended to system instructions for every LLM call in agent mode.
    """

    return "\n\n".join(
        [
            "### Function calling and JSON output",
            AGENT_RESPONSE_JSON_CONTRACT.strip(),
            build_tools_section(registry),
        ]
    ).strip()

