"""
MCP (Model Context Protocol) Integration Module

Implements the MCP 2025-11-25 specification for tool-calling:
https://modelcontextprotocol.io/specification/2025-11-25

Components:
- schema: MCP data types (tools, content blocks, JSON-RPC messages, initialize)
- tools: Tool registry with built-in tools and MCP tools/list + tools/call
- server: MCP server over JSON-RPC stdio transport
- client: MCP client for connecting to external MCP tool servers

All tool-calling in ARRG flows through this module using the MCP protocol.
"""

from .schema import (
    # JSON-RPC 2.0 transport
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCError,
    JSONRPCNotification,
    JSONRPC_VERSION,
    MCP_PROTOCOL_VERSION,
    # JSON-RPC error codes
    PARSE_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
    # Content types
    ContentType,
    TextContent,
    ImageContent,
    EmbeddedResource,
    Content,
    # Tool definitions
    MCPTool,
    MCPToolCall,
    MCPToolResult,
    # Initialize handshake
    MCPClientCapabilities,
    MCPServerCapabilities,
    MCPInitializeParams,
    MCPInitializeResult,
)

from .tools import (
    MCPToolRegistry,
    get_tool_registry,
    get_available_tools,
)

from .server import MCPServer, run_server
from .client import MCPClient

__all__ = [
    # JSON-RPC transport
    "JSONRPCRequest",
    "JSONRPCResponse",
    "JSONRPCError",
    "JSONRPCNotification",
    "JSONRPC_VERSION",
    "MCP_PROTOCOL_VERSION",
    # Error codes
    "PARSE_ERROR",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "INVALID_PARAMS",
    "INTERNAL_ERROR",
    # Content types
    "ContentType",
    "TextContent",
    "ImageContent",
    "EmbeddedResource",
    "Content",
    # Tool definitions
    "MCPTool",
    "MCPToolCall",
    "MCPToolResult",
    # Initialize
    "MCPClientCapabilities",
    "MCPServerCapabilities",
    "MCPInitializeParams",
    "MCPInitializeResult",
    # Tool registry
    "MCPToolRegistry",
    "get_tool_registry",
    "get_available_tools",
    # Server and client
    "MCPServer",
    "run_server",
    "MCPClient",
]
