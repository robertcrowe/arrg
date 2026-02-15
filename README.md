# ARRG - Automated Research Report Generator

A multi-agent system for generating comprehensive research reports using specialized AI agents communicating exclusively via the **A2A (Agent-to-Agent) Protocol v1.0** and using the **MCP (Model Context Protocol) 2025-11-25** specification for all tool-calling.

## Status

ARRG is currently well along, but still a work in progress. So far I have only tested with the Tetrate
provider, using `claude-haiku-4-5`. Example reports are in the `example_reports` directory.

## Overview

ARRG uses five specialized agents working together to produce high-quality research reports:

- **Planning Agent**: Creates structured research plans with outlines and methodologies
- **Research Agent**: Gathers information and sources based on research questions (uses MCP tools)
- **Analysis Agent**: Synthesizes research data into insights and findings
- **Writing Agent**: Transforms analysis into polished, professional reports
- **QA Agent**: Reviews and validates reports for quality and accuracy

### Protocol Architecture

ARRG uses two complementary protocols that separate concerns cleanly:

#### A2A Protocol v1.0 — Agent-to-Agent Communication

All agent-to-agent communication uses the [A2A Protocol](https://agent2agent.info/specification/) exclusively:

- **AgentCards** — Each agent advertises its capabilities via an `AgentCard` with skills, supported input/output modes, and provider metadata (per the A2A discovery spec)
- **Tasks** — The unit of work exchanged between agents. Tasks follow the A2A lifecycle state machine: `submitted → working → completed/failed` (with additional states: `canceled`, `input_required`, `rejected`, `auth_required`)
- **Messages** — Communication within tasks uses `Message` objects with a `role` (user/agent) and typed `Parts`:
  - `TextPart` — Plain text content
  - `DataPart` — Structured JSON data (research plans, analysis results, etc.)
  - `FilePart` — Binary file content with MIME types
- **Artifacts** — Deliverable outputs attached to completed tasks (reports, QA results, etc.)
- **TaskStatus** — Tracks state transitions with timestamps and status messages

The orchestrator creates A2A Tasks for each phase of the workflow and sends Messages to agents via `process_task()`. Agents process the task, transition through states, and return completed tasks with artifacts.

#### MCP 2025-11-25 — Tool Calling

All tool-calling follows the [MCP specification](https://spec.modelcontextprotocol.io/2025-11-25/):

- **Tool Discovery** — Tools are registered in an `MCPToolRegistry` with JSON Schema input definitions
- **Tool Invocation** — LLM-initiated tool calls are executed via `MCPToolCall` / `MCPToolResult`
- **Built-in Tools** — `web_search` for internet research, extensible to more tools
- **JSON-RPC 2.0** — MCP communication uses standard JSON-RPC over stdio transport

> **Why two protocols?** A2A handles *agent-to-agent* communication (task delegation, results, artifacts), while MCP handles *agent-to-tool* communication (search, file I/O, APIs). They are complementary — an agent receives work via A2A and uses MCP tools to accomplish it.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Orchestrator                       │
│         (creates A2A Tasks, routes Messages)         │
└─────────┬───────┬───────┬───────┬───────┬───────────┘
          │ A2A   │ A2A   │ A2A   │ A2A   │ A2A
          ▼       ▼       ▼       ▼       ▼
    ┌─────────┐ ┌─────┐ ┌────────┐ ┌───────┐ ┌────┐
    │Planning │ │Rsrch│ │Analysis│ │Writing│ │ QA │
    │ Agent   │ │Agent│ │ Agent  │ │ Agent │ │Agnt│
    └─────────┘ └──┬──┘ └────────┘ └───────┘ └────┘
                   │ MCP
                   ▼
              ┌─────────┐
              │MCP Tools│
              │(search) │
              └─────────┘
```

### Workflow

1. **Planning** — Orchestrator sends topic via A2A Task → Planning Agent returns research plan
2. **Research** — Research questions sent via A2A Task → Research Agent uses MCP `web_search` tool → returns findings
3. **Analysis** — Research data sent via A2A Task → Analysis Agent returns synthesized insights
4. **Writing** — Analysis sent via A2A Task → Writing Agent returns formatted report
5. **QA** — Report sent via A2A Task → QA Agent returns quality assessment
6. **Revision Loop** — If QA rejects, the report is sent back to Writing Agent with feedback (up to 2 retries)

### A2A Task Lifecycle

Each workflow phase follows the A2A task state machine:

```
submitted → working → completed
                   → failed
                   → input_required (for revision requests)
```

The orchestrator tracks all tasks and their state transitions, maintaining a full message history for debugging and audit.

## A2A Data Structures

### AgentCard

```python
from arrg.a2a import AgentCard, AgentSkill, AgentProvider, AgentCapabilities

card = AgentCard(
    name="Research Agent",
    description="Gathers information from web sources",
    url="local://research-agent",
    provider=AgentProvider(organization="ARRG"),
    capabilities=AgentCapabilities(streaming=True),
    skills=[
        AgentSkill(
            id="web_research",
            name="Web Research",
            description="Search and synthesize web sources",
            tags=["research", "search"],
        )
    ],
    input_modes=["application/json"],
    output_modes=["application/json"],
)
```

### Task & Message

```python
from arrg.a2a import Task, TaskState, Message, MessageRole, TextPart, DataPart

# Create a task
task = Task()

# Add a user message with text and structured data
message = Message.create_user_message(
    text="Research the impact of AI on healthcare",
    data={"topic": "AI in healthcare", "questions": ["What are the benefits?"]},
    sender="orchestrator",
    task_id=task.id,
)
task.add_message(message)

# Update task state
task.update_state(TaskState.WORKING, message="Agent processing...")
task.update_state(TaskState.COMPLETED, message="Research complete")
```

### Artifact

```python
from arrg.a2a import Artifact

# Create a data artifact
artifact = Artifact.create_data_artifact(
    data={"findings": [...], "sources": [...]},
    name="research_results",
    description="Research findings for AI in healthcare",
)
task.add_artifact(artifact)
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd arrg

# Install in development mode
pip install -e .

# Run the dashboard
python -m arrg dashboard

# Or generate a report from CLI
python -m arrg cli --topic "Your Research Topic" --api-key YOUR_API_KEY
```

## Usage

### Streamlit Dashboard

```bash
python -m arrg dashboard
# or
streamlit run arrg/ui/dashboard.py
```

### CLI

```bash
python -m arrg cli --topic "The Impact of AI on Healthcare" --api-key YOUR_KEY --model claude-haiku-4-5
```

### Programmatic

```python
from arrg import Orchestrator
from pathlib import Path

orchestrator = Orchestrator(
    api_key="your-api-key",
    provider_endpoint="Tetrate",
    models={"planning": "claude-haiku-4-5", "research": "claude-haiku-4-5", ...},
    workspace_dir=Path("./workspace"),
)

result = orchestrator.generate_report("Your Research Topic")
if result["status"] == "success":
    report = result["report"]
    print(report["title"])
    print(report["full_text"])
```

## Project Structure

```
arrg/
├── a2a/                    # A2A Protocol v1.0 implementation
│   ├── __init__.py         # Package exports
│   ├── agent_card.py       # AgentCard, AgentSkill, AgentProvider, AgentCapabilities
│   ├── task.py             # Task, TaskState, TaskStatus
│   ├── message.py          # Message, MessageRole, TextPart, DataPart, FilePart
│   └── artifact.py         # Artifact with typed Parts
├── agents/                 # Agent implementations
│   ├── base.py             # BaseAgent (abstract) - A2A + MCP integration
│   ├── planning.py         # PlanningAgent - research plan generation
│   ├── research.py         # ResearchAgent - web research via MCP tools
│   ├── analysis.py         # AnalysisAgent - data synthesis
│   ├── writing.py          # WritingAgent - report composition + revision
│   └── qa.py               # QAAgent - quality validation
├── core/
│   └── orchestrator.py     # Workflow orchestrator (A2A task coordination)
├── mcp/                    # MCP 2025-11-25 implementation
│   ├── client.py           # MCP client (JSON-RPC/stdio)
│   ├── server.py           # MCP server
│   ├── schema.py           # MCP tool schemas
│   └── tools.py            # Built-in tools (web_search)
├── protocol/               # Backward-compatible re-exports from a2a/
│   ├── __init__.py         # Re-exports A2A types + SharedWorkspace
│   ├── message.py          # Deprecated shim → arrg.a2a
│   └── workspace.py        # SharedWorkspace (key-value artifact storage)
├── ui/
│   └── dashboard.py        # Streamlit dashboard
├── utils/
│   └── llm_client.py       # LLM API client (OpenAI/Anthropic)
├── __init__.py             # Package exports
└── __main__.py             # CLI entry point
```

## Key Design Decisions

### A2A Protocol for All Agent Communication

Every interaction between agents uses A2A Protocol data structures:

1. **No custom message types** — All messages use `Message` with `TextPart`/`DataPart` instead of custom enums
2. **Task-centric workflow** — Each phase creates an A2A `Task` with proper state transitions
3. **Artifacts for outputs** — Agent deliverables are represented as A2A `Artifact` objects
4. **AgentCards for discovery** — Each agent advertises capabilities via an `AgentCard` with skills

### MCP for Tool Calling (Complementary)

MCP is used exclusively for tool calling (web search, etc.) — it does NOT handle agent-to-agent communication. This clean separation follows the intended design of both protocols:

- **A2A** = how agents talk to each other
- **MCP** = how agents use tools

### SharedWorkspace for Large Artifacts

Large data (research results, full reports) is stored in a `SharedWorkspace` and referenced by key in A2A messages. This avoids passing large payloads directly in messages while maintaining A2A Protocol compliance — the workspace key is passed as a `DataPart` within an A2A `Message`.

## Extending ARRG

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
2. Implement `get_capabilities()` returning a list of skill descriptions
3. Implement `process_task(task: Task) -> Task` following the A2A task lifecycle
4. Add the agent to the orchestrator workflow
5. Create an `AgentSkill` for each capability the agent offers

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
- [ ] HTTP JSON-RPC transport for true networked A2A communication
- [ ] A2A Agent Card serving at `/.well-known/agent-card.json`
- [ ] SSE streaming for real-time task updates via A2A
- [ ] A2A push notifications
- [ ] Replace mock tool executors with real implementations (web search API, file I/O, etc.)
- [ ] Support for multi-topic batch processing
- [ ] Advanced citation and reference management
- [ ] Integration with external research databases via MCP client connections
- [ ] Custom agent plugins
- [ ] Report templates and styling options
- [ ] Collaboration features for team workflows
- [ ] API endpoint for programmatic access
- [ ] MCP server discovery and multi-server tool aggregation
