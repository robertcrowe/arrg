# ARRG - Automated Research Report Generator

A multi-agent system for generating comprehensive research reports using specialized AI agents communicating via the A2A Protocol and using the **MCP (Model Context Protocol) 2025-11-25** specification for all tool-calling.

## Status

ARRG is currently well along, but still a work in progress.  So far I have only tested with the Tetrate
provider, using `claude-haiku-4-5`.  Example reports are in the `example_reports` directory.

## Overview

ARRG uses five specialized agents working together to produce high-quality research reports:

- **Planning Agent**: Creates structured research plans with outlines and methodologies
- **Research Agent**: Gathers information and sources based on research questions (uses MCP tools)
- **Analysis Agent**: Synthesizes research data into insights and findings
- **Writing Agent**: Transforms analysis into polished, professional reports
- **QA Agent**: Reviews and validates reports for quality and accuracy

All agents communicate using the **A2A (Agent-to-Agent) Protocol**, enabling standardized message passing and shared workspace access. All tool-calling follows the **MCP (Model Context Protocol) 2025-11-25 specification**, providing a standardized interface for tool discovery, invocation, and result handling.

## Features

### MCP Tool Integration (2025-11-25 Spec)
- **JSON-RPC 2.0 Transport**: Full MCP protocol over stdio with proper request/response framing
- **MCP Server**: Exposes ARRG's built-in tools (web_search, file_read, file_write, analyze_data, fact_check) as an MCP server
- **MCP Client**: Connects to external MCP servers for tool discovery and invocation
- **Agentic Tool-Call Loop**: Agents execute multi-turn tool-call → result cycles with configurable round limits
- **Initialize Handshake**: Proper MCP initialize/initialized lifecycle with capability negotiation
- **Typed Content**: MCP content types (TextContent, ImageContent, EmbeddedResource) for rich tool results

### Dashboard Interface (Streamlit)
- **Model Configuration**: Select LLM provider (Tetrate, OpenAI, Anthropic, Local) and model
- **Live Streaming Console**: Real-time output from all agents
- **Progress Tracking**: Visual workflow status for all 5 phases
- **Report Display**: Formatted view of generated reports with QA results
- **Export Options**: Download reports in Markdown, JSON, or TXT formats

### Command Line Interface
- Generate reports directly from the terminal
- Stream progress to console
- Save reports to files

## Installation

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd arrg
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

This will install ARRG along with its dependencies:
- **streamlit**: Dashboard UI framework
- **reportlab**: PDF export functionality
- **openai**: OpenAI and compatible API client
- **anthropic**: Anthropic API client

### API Configuration

ARRG requires an API key from one of the supported providers:

- **Tetrate**: Set `TETRATE_API_BASE` environment variable if using a custom endpoint
- **OpenAI**: Standard OpenAI API key
- **Anthropic**: Anthropic API key
- **Local**: Configure `LOCAL_API_BASE` for local model endpoints (e.g., Ollama)

You can enter the API key directly in the dashboard sidebar or pass it via the CLI.

## Usage

### Dashboard Mode (Default)

Launch the Streamlit dashboard:

```bash
python main.py
```

Or use the module directly:

```bash
python -m arrg dashboard
```

Or via Streamlit:

```bash
streamlit run arrg/ui/dashboard.py
```

**Using the Dashboard:**

1. Configure your settings in the sidebar:
   - Select your LLM provider
   - Choose a model
   - Enter your API key
   - Enable/disable options

2. Enter your research topic in the main input field

3. Click "Generate Report" to start the workflow

4. Monitor progress through:
   - Progress tracker (shows status of each phase)
   - Live console (streams agent output)

5. View the generated report with QA results

6. Export the report in your preferred format

### CLI Mode

Generate a report from the command line:

```bash
python -m arrg cli --topic "The Impact of AI on Healthcare" --api-key "your-api-key" --model "claude-haiku-4-5" --output report.md
```

Parameters:
- `--topic`: Research topic (required)
- `--api-key`: API key for the model provider (required)
- `--model`: Model to use (default: claude-haiku-4-5)
- `--provider`: Provider endpoint (default: Tetrate)
- `--output`: Output file path (optional)

### MCP Server Mode

Run ARRG's tools as a standalone MCP server (for use with any MCP client):

```bash
python -c "from arrg.mcp import MCPServer; MCPServer().run()"
```

This exposes the following tools over JSON-RPC 2.0 stdio transport:
- `web_search` - Search the web for information
- `file_read` - Read file contents
- `file_write` - Write content to files
- `analyze_data` - Analyze structured data
- `fact_check` - Verify claims against sources

### Check Version

```bash
python -m arrg version
```

## Architecture

### MCP (Model Context Protocol) — Tool Calling

All tool-calling in ARRG follows the **MCP 2025-11-25 specification** (<https://modelcontextprotocol.io>):

**Protocol Layer (JSON-RPC 2.0):**
- `initialize` / `initialized` — Capability negotiation handshake
- `tools/list` — Discover available tools with JSON Schema input definitions
- `tools/call` — Invoke a tool by name with validated arguments
- `ping` — Health check

**Key Types:**
- `MCPTool` — Tool definition with `name`, `description`, and `inputSchema` (JSON Schema)
- `MCPToolCall` — Tool invocation with `name`, `arguments`, and `call_id`
- `MCPToolResult` — Tool result with typed `content` array (TextContent, ImageContent, EmbeddedResource) and `is_error` flag
- `JSONRPCRequest` / `JSONRPCResponse` / `JSONRPCError` — JSON-RPC 2.0 message types

**Agentic Tool-Call Loop:**

When an agent calls `call_llm(use_tools=True)`, the following loop executes:

1. Tool schemas are retrieved from `MCPToolRegistry.get_tools_for_llm()` and sent with the LLM request
2. If the LLM response contains `tool_calls`, each call is:
   - Converted to an `MCPToolCall` 
   - Executed via `MCPToolRegistry.call_tool()` → `MCPToolResult`
   - Converted back to LLM message format via `MCPToolResult.to_llm_tool_result()`
3. Tool results are appended to the conversation and the LLM is called again
4. This repeats until the LLM returns plain text or the round limit is reached (default: 5)

**Components:**
- `arrg/mcp/schema.py` — MCP 2025-11-25 data types and JSON-RPC messages
- `arrg/mcp/tools.py` — Tool registry with built-in tool executors
- `arrg/mcp/server.py` — MCP server (JSON-RPC over stdio)
- `arrg/mcp/client.py` — MCP client for connecting to external MCP servers

### A2A Protocol — Agent Communication

The A2A Protocol enables standardized communication between agents:

**Message Types:**
- `TASK_REQUEST`: Request an agent to perform work
- `TASK_COMPLETE`: Notify that work is complete
- `CAPABILITY_QUERY`: Ask about agent capabilities
- `CAPABILITY_RESPONSE`: Respond with capabilities
- `ERROR`: Report an error

**Shared Workspace:**
- Agents store artifacts (plans, research, analysis, reports) in a shared workspace
- Messages pass references (keys) instead of large data payloads
- Supports both in-memory and persistent disk storage

### Workflow

1. **Planning Phase**: Planning Agent creates a structured research plan
2. **Research Phase**: Research Agent gathers information based on research questions (using MCP tools)
3. **Analysis Phase**: Analysis Agent synthesizes research into insights
4. **Writing Phase**: Writing Agent produces a polished report
5. **QA Phase**: QA Agent reviews the report and provides quality assessment

If the QA Agent rejects the report, the Writing Agent revises it (up to 2 retries).

### Project Structure

```
arrg/
├── agents/              # Specialized agents
│   ├── base.py         # Base agent class with MCP tool-call loop
│   ├── planning.py     # Planning agent
│   ├── research.py     # Research agent (uses MCP tools)
│   ├── analysis.py     # Analysis agent
│   ├── writing.py      # Writing agent
│   └── qa.py           # QA agent
├── mcp/                # MCP (Model Context Protocol) 2025-11-25
│   ├── schema.py       # MCP data types, JSON-RPC messages, content types
│   ├── tools.py        # Tool registry with built-in tool executors
│   ├── server.py       # MCP server (JSON-RPC over stdio)
│   └── client.py       # MCP client for external servers
├── protocol/           # A2A Protocol implementation
│   ├── message.py      # Message types and structures
│   └── workspace.py    # Shared workspace
├── a2a/                # A2A Protocol v1.0 data types
│   ├── agentcard.py    # Agent capabilities and metadata
│   ├── task.py         # Task lifecycle management
│   ├── message.py      # A2A message types
│   └── artifact.py     # Agent output artifacts
├── core/               # Core orchestration
│   └── orchestrator.py # Workflow coordinator
├── utils/              # Utilities
│   └── llm_client.py   # Multi-provider LLM client with tool-call support
├── ui/                 # User interface
│   └── dashboard.py    # Streamlit dashboard
└── __main__.py         # CLI entry point
```

## Configuration

### Supported Providers

- **Tetrate**: Access to multiple models (GPT-4o, Claude, O1, etc.)
- **OpenAI**: Direct OpenAI API access
- **Anthropic**: Direct Anthropic API access
- **Local**: Local model endpoints

### Supported Models

- GPT-4o, GPT-4o-mini
- O1, O1-mini
- Claude 3.5 Sonnet, Claude 3.5 Haiku
- Custom local models

## Development

### Running Tests

```bash
pytest test_*.py
```

### Code Structure

Each agent inherits from `BaseAgent` and implements:
- `get_capabilities()`: Return agent capabilities
- `process_message()`: Handle incoming A2A messages
- Custom methods for agent-specific logic

The `BaseAgent.call_llm()` method provides:
- Automatic MCP tool schema injection when `use_tools=True`
- Multi-turn agentic tool-call execution loop
- Configurable `max_tool_rounds` (default: 5)
- Provider-agnostic tool-call handling (OpenAI, Anthropic, Mock)

The orchestrator manages the workflow by:
1. Creating all specialized agents (each with its own MCP tool registry)
2. Sending task requests in sequence via A2A messages
3. Handling responses and errors
4. Coordinating the shared workspace

### Adding New MCP Tools

Register a new tool in `arrg/mcp/tools.py`:

```python
from arrg.mcp import MCPTool, MCPToolCall, MCPToolResult, TextContent

def my_tool_executor(call: MCPToolCall) -> MCPToolResult:
    """Execute the tool and return MCP-compliant result."""
    result_text = f"Executed with args: {call.arguments}"
    return MCPToolResult(
        call_id=call.call_id,
        content=[TextContent(text=result_text)],
        is_error=False,
    )

# In MCPToolRegistry._register_builtin_tools():
self.register_tool(
    MCPTool(
        name="my_tool",
        description="Description of my tool",
        inputSchema={
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"},
            },
            "required": ["param1"],
        },
    ),
    executor=my_tool_executor,
)
```

### Adding New Agents

1. Create a new agent class inheriting from `BaseAgent`
2. Implement required methods
3. Add the agent to the orchestrator
4. Update the workflow sequence

## Troubleshooting

### Common Issues

**API Key Error:**
- Ensure your API key is correctly entered in the sidebar
- Verify the key has access to the selected model

**Model Not Available:**
- Check that your provider supports the selected model
- Try a different model from the dropdown

**Workspace Errors:**
- Ensure the `./workspace` directory is writable
- Clear the workspace if it contains corrupted data

**Import Errors:**
- Verify the package is installed: `pip install -e .`
- Check Python version: `python --version` (requires 3.12+)

**MCP Tool Errors:**
- Check tool registry initialization: `python -c "from arrg.mcp import get_tool_registry; r = get_tool_registry(); print(r.list_tools())"`
- Verify tool schemas: `python -c "from arrg.mcp import get_tool_registry; r = get_tool_registry(); print([t.name for t in r.list_tools()])"`

## License

[Your License Here]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review the PRD (Product Requirements Document)

## Roadmap

Future enhancements:
- [ ] Replace mock tool executors with real implementations (web search API, file I/O, etc.)
- [ ] Support for multi-topic batch processing
- [ ] Advanced citation and reference management
- [ ] Integration with external research databases via MCP client connections
- [ ] Custom agent plugins
- [ ] Report templates and styling options
- [ ] Collaboration features for team workflows
- [ ] API endpoint for programmatic access
- [ ] MCP server discovery and multi-server tool aggregation
