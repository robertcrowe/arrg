"""
A2A (Agent-to-Agent) Protocol v1.0 implementation.

This module implements the A2A Protocol for agent-to-agent communication
as specified at https://agent2agent.info/specification/

Core data structures:
- AgentCard: Agent capability advertisement and discovery
- Task: Unit of work with lifecycle state management
- Message: Communication unit with typed Parts (text, data, file)
- Artifact: Deliverable output produced during task execution

See: https://github.com/google/A2A/blob/main/specification/a2a.proto
"""

from .agent_card import AgentCard, AgentProvider, AgentCapabilities, AgentSkill
from .task import Task, TaskState, TaskStatus
from .message import Message, MessageRole, TextPart, DataPart, FilePart, Part, part_from_dict
from .artifact import Artifact

__all__ = [
    # Agent discovery
    "AgentCard",
    "AgentProvider",
    "AgentCapabilities",
    "AgentSkill",
    # Task lifecycle
    "Task",
    "TaskState",
    "TaskStatus",
    # Messages and Parts
    "Message",
    "MessageRole",
    "TextPart",
    "DataPart",
    "FilePart",
    "Part",
    "part_from_dict",
    # Artifacts
    "Artifact",
]
