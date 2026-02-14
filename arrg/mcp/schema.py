"""
MCP (Model Context Protocol) Schema Definitions

Implements data structures following the MCP 2025-11-25 specification:
https://modelcontextprotocol.io/specification/2025-11-25

Key MCP concepts implemented:
- Tool definitions with JSON Schema inputSchema
- Tool call requests and results using content blocks
- JSON-RPC 2.0 message wrappers for protocol transport
- Typed content blocks (TextContent, ImageContent, EmbeddedResource)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import uuid
import json


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 base types (MCP transport layer)
# ---------------------------------------------------------------------------

JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2025-11-25"


@dataclass
class JSONRPCRequest:
    """
    JSON-RPC 2.0 request message.

    Per MCP spec, all messages are JSON-RPC 2.0.
    Requests have a string|number `id` and expect a response.
    """
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "method": self.method,
            "id": self.id,
        }
        if self.params is not None:
            msg["params"] = self.params
        return msg

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class JSONRPCNotification:
    """
    JSON-RPC 2.0 notification (no id, no response expected).
    """
    method: str
    params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "method": self.method,
        }
        if self.params is not None:
            msg["params"] = self.params
        return msg

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class JSONRPCResponse:
    """
    JSON-RPC 2.0 success response.
    """
    result: Any
    id: Optional[Union[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "result": self.result,
            "id": self.id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class JSONRPCError:
    """
    JSON-RPC 2.0 error response.
    """
    code: int
    message: str
    data: Optional[Any] = None
    id: Optional[Union[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        error_obj: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.data is not None:
            error_obj["data"] = self.data
        return {
            "jsonrpc": JSONRPC_VERSION,
            "error": error_obj,
            "id": self.id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# ---------------------------------------------------------------------------
# MCP Content types (used in tool results and resource contents)
# ---------------------------------------------------------------------------

class ContentType(str, Enum):
    """Content block types per MCP spec."""
    TEXT = "text"
    IMAGE = "image"
    RESOURCE = "resource"


@dataclass
class TextContent:
    """
    Text content block.

    MCP spec: { type: "text", text: string, annotations?: { ... } }
    """
    text: str
    annotations: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": ContentType.TEXT.value,
            "text": self.text,
        }
        if self.annotations:
            result["annotations"] = self.annotations
        return result


@dataclass
class ImageContent:
    """
    Image content block (base64-encoded).

    MCP spec: { type: "image", data: string, mimeType: string, annotations?: { ... } }
    """
    data: str  # base64-encoded image data
    mime_type: str  # e.g. "image/png"
    annotations: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": ContentType.IMAGE.value,
            "data": self.data,
            "mimeType": self.mime_type,
        }
        if self.annotations:
            result["annotations"] = self.annotations
        return result


@dataclass
class EmbeddedResource:
    """
    Embedded resource content block.

    MCP spec: { type: "resource", resource: { uri: string, text?: string, blob?: string, mimeType?: string } }
    """
    uri: str
    text: Optional[str] = None
    blob: Optional[str] = None  # base64-encoded
    mime_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        resource: Dict[str, Any] = {"uri": self.uri}
        if self.text is not None:
            resource["text"] = self.text
        if self.blob is not None:
            resource["blob"] = self.blob
        if self.mime_type is not None:
            resource["mimeType"] = self.mime_type
        return {
            "type": ContentType.RESOURCE.value,
            "resource": resource,
        }


# Union type for content blocks
Content = Union[TextContent, ImageContent, EmbeddedResource]


# ---------------------------------------------------------------------------
# MCP Tool definitions
# ---------------------------------------------------------------------------

@dataclass
class MCPTool:
    """
    MCP Tool definition following the 2025-11-25 specification.

    A tool has:
    - name: unique identifier
    - description: human-readable description (optional but recommended)
    - inputSchema: JSON Schema object describing the tool's parameters

    This is returned by `tools/list` and used by LLMs to decide which
    tools to call and with what arguments.
    """
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {},
    })
    annotations: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP tools/list response item format."""
        result: Dict[str, Any] = {
            "name": self.name,
            "inputSchema": self.input_schema,
        }
        if self.description:
            result["description"] = self.description
        if self.annotations:
            result["annotations"] = self.annotations
        return result

    def to_llm_format(self) -> Dict[str, Any]:
        """
        Convert to OpenAI-compatible function-calling format for LLM consumption.

        This is a convenience method for sending tool definitions to LLMs that
        use the OpenAI function-calling convention. The canonical MCP format
        is `to_dict()`.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            }
        }


# ---------------------------------------------------------------------------
# MCP tools/call request and result
# ---------------------------------------------------------------------------

@dataclass
class MCPToolCall:
    """
    Represents a tools/call request per the MCP specification.

    MCP spec `tools/call` params:
    {
        name: string,
        arguments?: { [key: string]: unknown }
    }

    The `call_id` is not part of the MCP spec itself but is used to
    correlate with LLM tool_call IDs when bridging between MCP and
    LLM provider APIs.
    """
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: Optional[str] = None

    def to_mcp_params(self) -> Dict[str, Any]:
        """Convert to MCP tools/call request params."""
        result: Dict[str, Any] = {"name": self.name}
        if self.arguments:
            result["arguments"] = self.arguments
        return result

    def to_jsonrpc_request(self, request_id: Optional[Union[str, int]] = None) -> JSONRPCRequest:
        """Wrap as a JSON-RPC 2.0 request for tools/call."""
        return JSONRPCRequest(
            method="tools/call",
            params=self.to_mcp_params(),
            id=request_id or str(uuid.uuid4()),
        )


@dataclass
class MCPToolResult:
    """
    Result of a tools/call per the MCP specification.

    MCP spec tools/call result:
    {
        content: (TextContent | ImageContent | EmbeddedResource)[],
        isError?: boolean
    }

    The `tool_name` and `call_id` fields are bookkeeping for correlation
    with the original call.
    """
    content: List[Content]
    is_error: bool = False
    tool_name: str = ""
    call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP tools/call result format."""
        result: Dict[str, Any] = {
            "content": [c.to_dict() for c in self.content],
        }
        if self.is_error:
            result["isError"] = True
        return result

    def to_jsonrpc_response(self, request_id: Optional[Union[str, int]] = None) -> JSONRPCResponse:
        """Wrap as a JSON-RPC 2.0 success response."""
        return JSONRPCResponse(
            result=self.to_dict(),
            id=request_id,
        )

    def get_text(self) -> str:
        """
        Convenience: concatenate all TextContent blocks into a single string.
        Useful for feeding tool results back to LLMs.
        """
        parts = []
        for block in self.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            elif isinstance(block, EmbeddedResource) and block.text:
                parts.append(block.text)
        return "\n".join(parts)

    def to_llm_tool_result(self) -> Dict[str, Any]:
        """
        Convert to OpenAI-compatible tool result message for LLM consumption.

        This bridges MCP tool results back into the LLM conversation format.
        """
        return {
            "role": "tool",
            "tool_call_id": self.call_id or "unknown",
            "content": self.get_text() if not self.is_error else f"Error: {self.get_text()}",
        }


# ---------------------------------------------------------------------------
# MCP Initialize handshake
# ---------------------------------------------------------------------------

@dataclass
class MCPClientCapabilities:
    """Client capabilities declared during initialize."""
    roots: Optional[Dict[str, Any]] = None
    sampling: Optional[Dict[str, Any]] = None
    experimental: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.roots is not None:
            result["roots"] = self.roots
        if self.sampling is not None:
            result["sampling"] = self.sampling
        if self.experimental is not None:
            result["experimental"] = self.experimental
        return result


@dataclass
class MCPServerCapabilities:
    """Server capabilities declared during initialize."""
    tools: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    experimental: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.tools is not None:
            result["tools"] = self.tools
        if self.resources is not None:
            result["resources"] = self.resources
        if self.prompts is not None:
            result["prompts"] = self.prompts
        if self.logging is not None:
            result["logging"] = self.logging
        if self.experimental is not None:
            result["experimental"] = self.experimental
        return result


@dataclass
class MCPInitializeParams:
    """Parameters for the initialize request (client → server)."""
    protocol_version: str = MCP_PROTOCOL_VERSION
    capabilities: MCPClientCapabilities = field(default_factory=MCPClientCapabilities)
    client_info: Dict[str, str] = field(default_factory=lambda: {
        "name": "arrg",
        "version": "0.1.0",
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities.to_dict(),
            "clientInfo": self.client_info,
        }


@dataclass
class MCPInitializeResult:
    """Result of the initialize request (server → client)."""
    protocol_version: str = MCP_PROTOCOL_VERSION
    capabilities: MCPServerCapabilities = field(default_factory=MCPServerCapabilities)
    server_info: Dict[str, str] = field(default_factory=lambda: {
        "name": "arrg-mcp-server",
        "version": "0.1.0",
    })
    instructions: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities.to_dict(),
            "serverInfo": self.server_info,
        }
        if self.instructions:
            result["instructions"] = self.instructions
        return result
