"""
MCP Client Implementation

Provides an MCP-compliant client that connects to MCP servers over
JSON-RPC 2.0 stdio transport, following the MCP 2025-11-25 specification:
https://modelcontextprotocol.io/specification/2025-11-25

The client handles:
1. initialize / initialized handshake
2. tools/list - discover server tools
3. tools/call - invoke server tools
4. ping - liveness check

Can connect to external MCP tool servers (e.g. filesystem, web search)
via subprocess stdio transport.
"""

import subprocess
import json
import logging
import uuid
from typing import Optional, Dict, Any, List

from .schema import (
    JSONRPCRequest,
    JSONRPCNotification,
    MCPInitializeParams,
    MCPTool,
    MCPToolCall,
    MCPToolResult,
    TextContent,
    MCP_PROTOCOL_VERSION,
    JSONRPC_VERSION,
)


logger = logging.getLogger("arrg.mcp.client")


class MCPClient:
    """
    MCP Client that connects to an MCP server over stdio transport.

    Usage:
        client = MCPClient(command=["python", "-m", "some_mcp_server"])
        client.connect()
        tools = client.list_tools()
        result = client.call_tool("web_search", {"query": "AI research"})
        client.disconnect()

    Or as a context manager:
        with MCPClient(command=["python", "-m", "some_mcp_server"]) as client:
            tools = client.list_tools()
    """

    def __init__(
        self,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the MCP client.

        Args:
            command: Command and arguments to launch the MCP server process.
            env: Optional environment variables for the server process.
            timeout: Timeout in seconds for server responses.
        """
        self.command = command
        self.env = env
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._initialized = False
        self._server_info: Optional[Dict[str, str]] = None
        self._server_capabilities: Optional[Dict[str, Any]] = None

    def __enter__(self) -> "MCPClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> Dict[str, Any]:
        """
        Start the server process and perform the MCP initialize handshake.

        Returns:
            The initialize result from the server.

        Raises:
            RuntimeError: If the server fails to start or handshake fails.
        """
        logger.info(f"Starting MCP server: {' '.join(self.command)}")

        try:
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env,
            )
        except (OSError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to start MCP server: {e}") from e

        # Send initialize request
        init_params = MCPInitializeParams()
        result = self._send_request("initialize", init_params.to_dict())

        self._server_info = result.get("serverInfo", {})
        self._server_capabilities = result.get("capabilities", {})
        protocol_version = result.get("protocolVersion", "unknown")

        logger.info(
            f"Connected to {self._server_info.get('name', 'unknown')} "
            f"(protocol {protocol_version})"
        )

        # Send initialized notification
        self._send_notification("notifications/initialized")
        self._initialized = True

        return result

    def disconnect(self) -> None:
        """Terminate the server process."""
        if self._process:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
                if self._process:
                    self._process.kill()
            finally:
                self._process = None
                self._initialized = False

        logger.info("MCP client disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected to a server."""
        return (
            self._process is not None
            and self._process.poll() is None
            and self._initialized
        )

    # ------------------------------------------------------------------
    # MCP operations
    # ------------------------------------------------------------------

    def list_tools(self) -> List[MCPTool]:
        """
        Discover available tools from the server (tools/list).

        Returns:
            List of MCPTool definitions.
        """
        self._ensure_connected()
        result = self._send_request("tools/list")
        tools = []
        for t in result.get("tools", []):
            tools.append(MCPTool(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
            ))
        return tools

    def call_tool(self, call: MCPToolCall) -> MCPToolResult:
        """
        Invoke a tool on the server (tools/call).

        Args:
            call: MCPToolCall with tool name and arguments.

        Returns:
            MCPToolResult with content blocks and error status.
        """
        self._ensure_connected()
        try:
            result = self._send_request("tools/call", call.to_mcp_params())
        except RuntimeError as e:
            return MCPToolResult(
                content=[TextContent(text=f"Server error: {e}")],
                is_error=True,
                tool_name=call.name,
                call_id=call.call_id,
            )

        # Parse MCP result
        content_blocks = []
        for block in result.get("content", []):
            block_type = block.get("type", "text")
            if block_type == "text":
                content_blocks.append(TextContent(text=block.get("text", "")))
            # ImageContent and EmbeddedResource could be parsed here too
            else:
                # Fallback: treat unknown types as text
                content_blocks.append(TextContent(text=json.dumps(block)))

        if not content_blocks:
            content_blocks = [TextContent(text="(empty result)")]

        return MCPToolResult(
            content=content_blocks,
            is_error=result.get("isError", False),
            tool_name=call.name,
            call_id=call.call_id,
        )

    def call_tool_simple(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> MCPToolResult:
        """
        Convenience: call a tool by name with arguments dict.
        """
        return self.call_tool(MCPToolCall(name=name, arguments=arguments or {}))

    def ping(self) -> bool:
        """Send a ping to check server liveness."""
        try:
            self._send_request("ping")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # JSON-RPC transport
    # ------------------------------------------------------------------

    def _send_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a JSON-RPC request and wait for the response.

        Returns:
            The result field from the JSON-RPC response.

        Raises:
            RuntimeError on transport or protocol errors.
        """
        self._ensure_process()

        request_id = str(uuid.uuid4())
        request = JSONRPCRequest(method=method, params=params, id=request_id)
        message = request.to_json() + "\n"

        try:
            self._process.stdin.write(message)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"Failed to send to MCP server: {e}") from e

        # Read response line
        try:
            response_line = self._process.stdout.readline()
        except Exception as e:
            raise RuntimeError(f"Failed to read from MCP server: {e}") from e

        if not response_line:
            raise RuntimeError("MCP server closed connection (empty response)")

        try:
            data = json.loads(response_line.strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from MCP server: {e}") from e

        # Check for JSON-RPC error
        if "error" in data:
            error = data["error"]
            raise RuntimeError(
                f"MCP server error [{error.get('code', '?')}]: "
                f"{error.get('message', 'unknown error')}"
            )

        return data.get("result", {})

    def _send_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        self._ensure_process()

        notification = JSONRPCNotification(method=method, params=params)
        message = notification.to_json() + "\n"

        try:
            self._process.stdin.write(message)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            logger.warning(f"Failed to send notification: {e}")

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if not self.is_connected:
            raise RuntimeError("MCP client is not connected. Call connect() first.")

    def _ensure_process(self) -> None:
        """Raise if process is not running."""
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("MCP server process is not running")
