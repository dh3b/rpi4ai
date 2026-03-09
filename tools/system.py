from __future__ import annotations

import platform
from datetime import datetime
from typing import Dict

from tools.registry import ToolRegistry, tool


def register_system_tools(registry: ToolRegistry) -> None:
    @tool(registry=registry)
    def get_time() -> str:
        """
        Get the current local date/time as an ISO-8601 string.

        Args: none
        Returns:
          ISO 8601 datetime string, e.g. "2026-03-09T12:34:56"
        """

        return datetime.now().replace(microsecond=0).isoformat()

    @tool(registry=registry)
    def get_platform_info() -> Dict[str, str]:
        """
        Get basic platform information (OS, machine, python version).

        Args: none
        Returns:
          Object with keys: system, release, version, machine, processor, python_version
        """

        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }

