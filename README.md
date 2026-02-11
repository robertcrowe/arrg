# ARRG - Automated Research Report Generator

A multi-agent system for generating comprehensive research reports using specialized AI agents communicating via the A2A Protocol.

## Status

ARRG is currently well along, but still a work in progress.  So far I have only tested with the Tetrate
provider, using Claude Sonnet 4.5 and Claude Opus 4.6.

## Overview

ARRG uses five specialized agents working together to produce high-quality research reports:

- **Planning Agent**: Creates structured research plans with outlines and methodologies
- **Research Agent**: Gathers information and sources based on research questions
- **Analysis Agent**: Synthesizes research data into insights and findings
- **Writing Agent**: Transforms analysis into polished, professional reports
- **QA Agent**: Reviews and validates reports for quality and accuracy

All agents communicate using the **A2A (Agent-to-Agent) Protocol**, enabling standardized message passing and shared workspace access.

## Features

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

### Check Version

```bash
python -m arrg version
```

## Architecture

### A2A Protocol

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
2. **Research Phase**: Research Agent gathers information based on research questions
3. **Analysis Phase**: Analysis Agent synthesizes research into insights
4. **Writing Phase**: Writing Agent produces a polished report
5. **QA Phase**: QA Agent reviews the report and provides quality assessment

### Project Structure

```
arrg/
├── agents/              # Specialized agents
│   ├── base.py         # Base agent class
│   ├── planning.py     # Planning agent
│   ├── research.py     # Research agent
│   ├── analysis.py     # Analysis agent
│   ├── writing.py      # Writing agent
│   └── qa.py           # QA agent
├── protocol/           # A2A Protocol implementation
│   ├── message.py      # Message types and structures
│   └── workspace.py    # Shared workspace
├── core/               # Core orchestration
│   └── orchestrator.py # Workflow coordinator
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
pytest tests/
```

### Code Structure

Each agent inherits from `BaseAgent` and implements:
- `get_capabilities()`: Return agent capabilities
- `process_message()`: Handle incoming A2A messages
- Custom methods for agent-specific logic

The orchestrator manages the workflow by:
1. Creating all specialized agents
2. Sending task requests in sequence
3. Handling responses and errors
4. Coordinating the shared workspace

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
- [ ] Support for multi-topic batch processing
- [ ] Advanced citation and reference management
- [ ] Integration with external research databases
- [ ] Custom agent plugins
- [ ] Report templates and styling options
- [ ] Collaboration features for team workflows
- [ ] API endpoint for programmatic access
