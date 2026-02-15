"""
Protocol module - backward compatibility layer.

All A2A Protocol types are now in arrg.a2a. This module re-exports them
for backward compatibility and also provides the SharedWorkspace utility
for artifact storage.

Primary types for new code should be imported from arrg.a2a directly.
"""

# A2A Protocol types (re-exported for backward compatibility)
from arrg.a2a import (
    Task,
    TaskState,
    TaskStatus,
    Message,
    MessageRole,
    TextPart,
    DataPart,
    FilePart,
    Artifact,
    AgentCard,
    AgentSkill,
    AgentProvider,
    AgentCapabilities,
)

# SharedWorkspace is local to protocol module
from .workspace import SharedWorkspace

__all__ = [
    # A2A Protocol types
    "Task",
    "TaskState",
    "TaskStatus",
    "Message",
    "MessageRole",
    "TextPart",
    "DataPart",
    "FilePart",
    "Artifact",
    "AgentCard",
    "AgentSkill",
    "AgentProvider",
    "AgentCapabilities",
    # Workspace
    "SharedWorkspace",
]
