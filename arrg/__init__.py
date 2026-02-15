"""
ARRG - Automated Research Report Generator.

A multi-agent system for generating comprehensive research reports.

Communication Architecture:
- A2A Protocol v1.0 for all agent-to-agent communication
  (Tasks, Messages with Parts, Artifacts, AgentCards)
- MCP 2025-11-25 for tool-calling (web_search, etc.)
  (complementary to A2A)

See: https://agent2agent.info/specification/
"""

__version__ = "0.1.0"

# Core orchestrator
from arrg.core import Orchestrator

# A2A Protocol types (primary communication protocol)
from arrg.a2a import (
    Task,
    TaskState,
    TaskStatus,
    Message,
    MessageRole,
    TextPart,
    DataPart,
    Artifact,
    AgentCard,
    AgentSkill,
    AgentProvider,
    AgentCapabilities,
)

# Workspace for artifact storage
from arrg.protocol import SharedWorkspace

# Agent classes
from arrg.agents import (
    BaseAgent,
    PlanningAgent,
    ResearchAgent,
    AnalysisAgent,
    WritingAgent,
    QAAgent,
)
