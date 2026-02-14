"""
MCP Server Implementation

Provides an MCP-compliant server that exposes ARRG's tools over JSON-RPC 2.0
transport (stdio), following the MCP 2025-11-25 specification:
https://modelcontextprotocol.io/specification/2025-11-25

The server handles the full MCP lifecycle:
1. initialize / initialized handshake
2. tools/list - enumerate available tools
3. tools/call - execute tool invocations
4. ping - liveness check
5. notifications/cancelled - cancellation support

Transport: Reads newline-delimited JSON-RPC messages from stdin, writes
responses to stdout (stdio transport per MCP spec).
"""

import sys
import json
import logging
from typing import Optional, Dict, Any

from .schema import (
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCError,
    JSONRPCNotification,
    MCPInitializeResult,
    MCPServerCapabilities,
    MCP_PROTOCOL_VERSION,
    JSONRPC_VERSION,
    PARSE_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    INTERNAL_ERROR,
)
from .tools import MCPToolRegistry, get_tool_registry


logger = logging.getLogger("arrg.mcp.server")


class MCPServer:
    """
    MCP Server that exposes tools over JSON-RPC 2.0 stdio transport.

    Usage:
        server = MCPServer()
        server.run()  # blocks, reading from stdin

    Or for programmatic use:
        server = MCPServer()
        response = server.handle_message(json_string)
    """

    def __init__(self, registry: Optional[MCPToolRegistry] = None):
        """
        Initialize the MCP server.

        Args:
            registry: Tool registry to serve.  Uses global registry if None.
        """
        self.registry = registry or get_tool_registry()
        self._initialized = False
        self._client_info: Optional[Dict[str, str]] = None

    # ------------------------------------------------------------------
    # Stdio transport
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Run the server, reading JSON-RPC messages from stdin and writing
        responses to stdout.  Blocks until stdin is closed or EOF.
        """
        logger.info("MCP Server starting on stdio transport")

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            response = self.handle_message(line)
            if response is not None:
                sys.stdout.write(response + "\n")
                sys.stdout.flush()

        logger.info("MCP Server shutting down (stdin closed)")

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def handle_message(self, raw: str) -> Optional[str]:
        """
        Parse and dispatch a single JSON-RPC message.

        Args:
            raw: Raw JSON string.

        Returns:
            JSON string response, or None for notifications.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            err = JSONRPCError(
                code=PARSE_ERROR,
                message=f"Parse error: {e}",
            )
            return err.to_json()

        # Validate basic JSON-RPC structure
        if not isinstance(data, dict):
            err = JSONRPCError(
                code=INVALID_REQUEST,
                message="Invalid request: expected JSON object",
            )
            return err.to_json()

        jsonrpc = data.get("jsonrpc")
        if jsonrpc != JSONRPC_VERSION:
            err = JSONRPCError(
                code=INVALID_REQUEST,
                message=f"Invalid JSON-RPC version: {jsonrpc}",
                id=data.get("id"),
            )
            return err.to_json()

        method = data.get("method")
        params = data.get("params", {})
        msg_id = data.get("id")

        # Notification (no id) – no response expected
        if msg_id is None:
            self._handle_notification(method, params)
            return None

        # Request (has id) – response required
        return self._handle_request(method, params, msg_id)

    # ------------------------------------------------------------------
    # Request dispatch
    # ------------------------------------------------------------------

    def _handle_request(
        self, method: str, params: Dict[str, Any], msg_id: Any
    ) -> str:
        """Dispatch a JSON-RPC request and return the response JSON."""
        try:
            if method == "initialize":
                return self._handle_initialize(params, msg_id)
            elif method == "ping":
                return JSONRPCResponse(result={}, id=msg_id).to_json()
            elif method == "tools/list":
                return self._handle_tools_list(params, msg_id)
            elif method == "tools/call":
                return self._handle_tools_call(params, msg_id)
            else:
                return JSONRPCError(
                    code=METHOD_NOT_FOUND,
                    message=f"Method not found: {method}",
                    id=msg_id,
                ).to_json()
        except Exception as e:
            logger.error(f"Internal error handling {method}: {e}", exc_info=True)
            return JSONRPCError(
                code=INTERNAL_ERROR,
                message=f"Internal error: {e}",
                id=msg_id,
            ).to_json()

    def _handle_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Handle a JSON-RPC notification (no response)."""
        if method == "notifications/initialized":
            logger.info("Client confirmed initialization")
            self._initialized = True
        elif method == "notifications/cancelled":
            request_id = params.get("requestId")
            reason = params.get("reason", "unknown")
            logger.info(f"Client cancelled request {request_id}: {reason}")
        else:
            logger.debug(f"Unhandled notification: {method}")

    # ------------------------------------------------------------------
    # MCP method handlers
    # ------------------------------------------------------------------

    def _handle_initialize(
        self, params: Dict[str, Any], msg_id: Any
    ) -> str:
        """Handle the initialize request."""
        self._client_info = params.get("clientInfo", {})
        client_version = params.get("protocolVersion", "unknown")
        logger.info(
            f"Initialize from {self._client_info.get('name', 'unknown')} "
            f"(protocol {client_version})"
        )

        result = MCPInitializeResult(
            capabilities=MCPServerCapabilities(
                tools={"listChanged": True},
            ),
        )
        return JSONRPCResponse(result=result.to_dict(), id=msg_id).to_json()

    def _handle_tools_list(
        self, params: Dict[str, Any], msg_id: Any
    ) -> str:
        """Handle tools/list request."""
        cursor = params.get("cursor")
        result = self.registry.list_tools(cursor=cursor)
        return JSONRPCResponse(result=result, id=msg_id).to_json()

    def _handle_tools_call(
        self, params: Dict[str, Any], msg_id: Any
    ) -> str:
        """Handle tools/call request."""
        from .schema import MCPToolCall

        name = params.get("name")
        if not name:
            return JSONRPCError(
                code=-32602,  # INVALID_PARAMS
                message="Missing required parameter: name",
                id=msg_id,
            ).to_json()

        arguments = params.get("arguments", {})
        call = MCPToolCall(name=name, arguments=arguments)
        tool_result = self.registry.call_tool(call)

        # MCP spec: tool execution errors are returned in the result
        # (isError=true), NOT as JSON-RPC errors.
        return JSONRPCResponse(
            result=tool_result.to_dict(), id=msg_id
        ).to_json()


def run_server() -> None:
    """Entry point for running the MCP server."""
    server = MCPServer()
    server.run()
