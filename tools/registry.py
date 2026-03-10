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
        """
        Invoke a tool by name with a dict of JSON arguments.

        Extra keys supplied by the LLM are ignored unless the tool explicitly
        accepts **kwargs. This makes tool-calling robust to hallucinated
        arguments such as unexpected keyword parameters.
        """
        spec = self.get(name)
        func = spec.func
        sig = inspect.signature(func)

        # If the tool accepts **kwargs, pass all arguments through as-is.
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_kw:
            return func(**(args or {}))

        # Otherwise, only pass arguments that match the function's parameter names.
        safe_kwargs: Dict[str, Any] = {}
        incoming = args or {}
        for param_name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL):
                continue
            if param_name in incoming:
                safe_kwargs[param_name] = incoming[param_name]

        return func(**safe_kwargs)


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

