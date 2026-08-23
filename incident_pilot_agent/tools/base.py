"""Plain internal Tool interface: name, description, typed Pydantic input
schema, execute(). Deliberately not MCP — there is no separate process
boundary here, only this codebase's own agents calling these tools. Shaped
so a future migration to MCP tool servers is mechanical: `name` maps to an
MCP tool name, `input_schema` maps to an MCP input schema (already JSON
Schema via Pydantic), and `execute()` maps to an MCP tool call handler.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Optional, Type

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Every tool call returns this — never raises for an expected failure
    mode (backend unavailable, no matching data). `data` is tool-specific
    structured output; `query_summary` is what gets recorded to the
    trajectory log (never the full raw payload)."""

    tool_name: str
    ok: bool
    data: Any = None
    error: Optional[str] = None
    query_summary: str


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[Type[BaseModel]]

    def spec(self) -> Dict[str, Any]:
        """JSON-schema tool spec, e.g. for Claude's native tool-use or a
        future MCP tool listing."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
        }

    @abstractmethod
    async def execute(self, tool_input: BaseModel) -> ToolResult: ...
