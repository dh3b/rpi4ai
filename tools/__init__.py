"""Tool registry and built-in agent tools."""

from tools.registry import ToolRegistry
from tools.system import register_system_tools
from tools.gpio import register_gpio_tools


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_system_tools(registry)
    register_gpio_tools(registry)
    return registry

