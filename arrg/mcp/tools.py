"""
MCP Tool Registry

Provides an MCP-compliant tool registry for the ARRG system following the
MCP 2025-11-25 specification:
https://modelcontextprotocol.io/specification/2025-11-25

Tools are defined with JSON Schema inputSchema and executed through the
standard `tools/call` interface.  Results are returned as typed content
blocks (TextContent, ImageContent, EmbeddedResource).

The registry can be used:
1. In-process – call `list_tools()` / `call_tool()` directly.
2. As an MCP server – wrap with `MCPServer` for JSON-RPC over stdio.
3. For LLM integration – `get_tools_for_llm()` converts to OpenAI format.
"""

from typing import Dict, List, Any, Callable, Optional
import logging

from .schema import (
    MCPTool,
    MCPToolCall,
    MCPToolResult,
    TextContent,
    Content,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCError,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)


logger = logging.getLogger("arrg.mcp.tools")


class MCPToolRegistry:
    """
    Registry for MCP-compliant tools.

    Manages tool definitions and execution following the Model Context Protocol
    2025-11-25 specification.  Tools are registered with an `MCPTool` schema
    and an executor callable, then discovered via `list_tools()` and invoked
    via `call_tool()`.
    """

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, MCPTool] = {}
        self._executors: Dict[str, Callable[..., str]] = {}
        self._register_builtin_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_tool(self, tool: MCPTool, executor: Callable[..., str]) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: MCP tool definition (name, description, inputSchema).
            executor: Callable that implements the tool.  It receives keyword
                      arguments matching the tool's inputSchema properties
                      and returns a plain string (which will be wrapped in a
                      TextContent block).
        """
        self._tools[tool.name] = tool
        self._executors[tool.name] = executor
        logger.info(f"Registered MCP tool: {tool.name}")

    def unregister_tool(self, name: str) -> bool:
        """Remove a tool from the registry. Returns True if it existed."""
        existed = name in self._tools
        self._tools.pop(name, None)
        self._executors.pop(name, None)
        return existed

    # ------------------------------------------------------------------
    # MCP tools/list
    # ------------------------------------------------------------------

    def list_tools(self, cursor: Optional[str] = None) -> Dict[str, Any]:
        """
        Return tools in MCP `tools/list` response format.

        MCP spec result: { tools: Tool[], nextCursor?: string }
        Pagination via cursor is accepted but not yet implemented (all
        tools are returned in a single page).
        """
        return {
            "tools": [tool.to_dict() for tool in self._tools.values()],
        }

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool definition by name, or None if not found."""
        return self._tools.get(name)

    # ------------------------------------------------------------------
    # MCP tools/call
    # ------------------------------------------------------------------

    def call_tool(self, call: MCPToolCall) -> MCPToolResult:
        """
        Execute a tool call per the MCP `tools/call` specification.

        Args:
            call: MCPToolCall with name and arguments.

        Returns:
            MCPToolResult with content blocks and isError flag.
        """
        if call.name not in self._executors:
            return MCPToolResult(
                content=[TextContent(text=f"Tool '{call.name}' not found")],
                is_error=True,
                tool_name=call.name,
                call_id=call.call_id,
            )

        try:
            executor = self._executors[call.name]
            text_result = executor(**call.arguments)
            return MCPToolResult(
                content=[TextContent(text=text_result)],
                is_error=False,
                tool_name=call.name,
                call_id=call.call_id,
            )
        except TypeError as e:
            # Argument mismatch
            logger.error(f"Invalid arguments for tool {call.name}: {e}")
            return MCPToolResult(
                content=[TextContent(text=f"Invalid arguments: {e}")],
                is_error=True,
                tool_name=call.name,
                call_id=call.call_id,
            )
        except Exception as e:
            logger.error(f"Error executing tool {call.name}: {e}")
            return MCPToolResult(
                content=[TextContent(text=f"Execution error: {e}")],
                is_error=True,
                tool_name=call.name,
                call_id=call.call_id,
            )

    # ------------------------------------------------------------------
    # JSON-RPC dispatch (for MCP server usage)
    # ------------------------------------------------------------------

    def handle_jsonrpc(self, request: JSONRPCRequest) -> JSONRPCResponse | JSONRPCError:
        """
        Dispatch a JSON-RPC request to the appropriate handler.

        Supports:
        - tools/list → list_tools()
        - tools/call → call_tool()
        """
        if request.method == "tools/list":
            cursor = (request.params or {}).get("cursor")
            result = self.list_tools(cursor=cursor)
            return JSONRPCResponse(result=result, id=request.id)

        elif request.method == "tools/call":
            params = request.params or {}
            name = params.get("name")
            if not name:
                return JSONRPCError(
                    code=INVALID_PARAMS,
                    message="Missing required parameter: name",
                    id=request.id,
                )
            arguments = params.get("arguments", {})
            call = MCPToolCall(name=name, arguments=arguments)
            tool_result = self.call_tool(call)

            # MCP spec: tool errors are returned as isError in the result,
            # not as JSON-RPC errors (those are for protocol-level failures).
            return JSONRPCResponse(result=tool_result.to_dict(), id=request.id)

        else:
            return JSONRPCError(
                code=METHOD_NOT_FOUND,
                message=f"Method not found: {request.method}",
                id=request.id,
            )

    # ------------------------------------------------------------------
    # LLM integration helpers
    # ------------------------------------------------------------------

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get tools in OpenAI function-calling format for LLM consumption.

        This is a convenience bridge: the canonical format is MCP's
        `tools/list`, but LLM providers expect OpenAI-style schemas.
        """
        return [tool.to_llm_format() for tool in self._tools.values()]

    # ------------------------------------------------------------------
    # Built-in tools
    # ------------------------------------------------------------------

    def _register_builtin_tools(self) -> None:
        """Register ARRG's built-in tools."""

        # Web Search
        self.register_tool(
            MCPTool(
                name="web_search",
                description="Search the web for information on a given query. Returns relevant search results.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            executor=self._mock_web_search,
        )

        # File Read
        self.register_tool(
            MCPTool(
                name="file_read",
                description="Read contents from a file in the workspace",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to read",
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            executor=self._mock_file_read,
        )

        # File Write
        self.register_tool(
            MCPTool(
                name="file_write",
                description="Write or update content to a file in the workspace",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to write",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            ),
            executor=self._mock_file_write,
        )

        # Data Analysis
        self.register_tool(
            MCPTool(
                name="analyze_data",
                description="Analyze data and extract insights, patterns, or statistics",
                input_schema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "The data to analyze (text, JSON, or structured format)",
                        },
                        "analysis_type": {
                            "type": "string",
                            "description": "Type of analysis to perform",
                            "enum": ["summary", "patterns", "statistics", "sentiment"],
                            "default": "summary",
                        },
                    },
                    "required": ["data"],
                },
            ),
            executor=self._mock_analyze_data,
        )

        # Fact Check
        self.register_tool(
            MCPTool(
                name="fact_check",
                description="Verify facts and claims against available sources",
                input_schema={
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "The claim or statement to verify",
                        },
                        "sources": {
                            "type": "string",
                            "description": "Sources to check against (optional)",
                        },
                    },
                    "required": ["claim"],
                },
            ),
            executor=self._mock_fact_check,
        )

    # ------------------------------------------------------------------
    # Mock tool executors (to be replaced with real implementations)
    # ------------------------------------------------------------------

    def _mock_web_search(self, query: str, max_results: int = 5) -> str:
        """Mock web search implementation."""
        logger.info(f"Mock web search: {query} (max_results={max_results})")
        return (
            f"Web search results for '{query}':\n\n"
            "1. Recent developments show significant progress in this area\n"
            "2. Multiple sources confirm growing interest and adoption\n"
            "3. Expert opinions highlight both opportunities and challenges\n"
            "4. Latest research indicates promising future directions\n"
            "5. Industry reports suggest continued growth and innovation\n\n"
            "Note: This is a mock result. Real implementation would use actual web search APIs."
        )

    def _mock_file_read(self, file_path: str) -> str:
        """Mock file read implementation."""
        logger.info(f"Mock file read: {file_path}")
        return f"[Mock file content from {file_path}]\n\nThis is sample content that would be read from the file."

    def _mock_file_write(self, file_path: str, content: str) -> str:
        """Mock file write implementation."""
        logger.info(f"Mock file write: {file_path} ({len(content)} chars)")
        return f"Successfully wrote {len(content)} characters to {file_path} (mock operation)"

    def _mock_analyze_data(self, data: str, analysis_type: str = "summary") -> str:
        """Mock data analysis implementation."""
        logger.info(f"Mock data analysis: type={analysis_type}, data_length={len(data)}")
        return (
            f"Data Analysis ({analysis_type}):\n\n"
            f"- Data size: {len(data)} characters\n"
            f"- Analysis type: {analysis_type}\n"
            "- Key insights: Multiple patterns detected\n"
            "- Summary: The data shows consistent themes across sources\n"
            "- Recommendation: Further investigation warranted\n\n"
            "Note: This is a mock analysis. Real implementation would perform actual data analysis."
        )

    def _mock_fact_check(self, claim: str, sources: str = None) -> str:
        """Mock fact checking implementation."""
        logger.info(f"Mock fact check: {claim}")
        return (
            "Fact Check Result:\n\n"
            f'Claim: "{claim}"\n\n'
            "Status: VERIFIED (mock)\n"
            "Confidence: Medium\n"
            f"Sources Checked: {sources if sources else 'Multiple reliable sources'}\n\n"
            "Summary: The claim appears to be generally accurate based on available information.\n"
            "This is a mock verification - actual fact checking would verify against real sources."
        )


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_registry: Optional[MCPToolRegistry] = None


def get_tool_registry() -> MCPToolRegistry:
    """Get the global tool registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = MCPToolRegistry()
    return _global_registry


def get_available_tools() -> List[MCPTool]:
    """Get list of all available MCP tools."""
    registry = get_tool_registry()
    return list(registry._tools.values())
