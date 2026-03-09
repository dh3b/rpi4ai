from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional


ToolFunc = Callable[..., Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    func: ToolFunc
    description: str
    signature: str


def _clean_docstring(doc: Optional[str]) -> str:
    if not doc:
        return ""
    return inspect.cleandoc(doc).strip()


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, func: ToolFunc, *, name: Optional[str] = None, description: Optional[str] = None) -> ToolFunc:
        tool_name = (name or func.__name__).strip()
        if not tool_name:
            raise ValueError("Tool name cannot be empty")
        if tool_name in self._tools:
            raise ValueError(f"Tool already registered: {tool_name}")

        sig = str(inspect.signature(func))
        doc = _clean_docstring(description or func.__doc__)
        spec = ToolSpec(
            name=tool_name,
            func=func,
            description=doc or "No description provided.",
            signature=f"{tool_name}{sig}",
        )
        self._tools[tool_name] = spec
        return func

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> Iterable[ToolSpec]:
        return (self._tools[k] for k in sorted(self._tools.keys()))

    def call(self, name: str, args: Dict[str, Any]) -> Any:
        spec = self.get(name)
        return spec.func(**(args or {}))


def tool(*, registry: ToolRegistry, name: Optional[str] = None, description: Optional[str] = None):
    """
    Decorator to register a function as an agent tool.

    Usage:
      registry = ToolRegistry()
      @tool(registry=registry)
      def get_time() -> str: ...
    """

    def _decorator(func: ToolFunc) -> ToolFunc:
        return registry.register(func, name=name, description=description)

    return _decorator

