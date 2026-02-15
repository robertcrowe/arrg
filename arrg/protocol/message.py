"""
DEPRECATED: Legacy protocol message module.

All A2A Protocol types have moved to arrg.a2a.
This module provides backward-compatible aliases only.

Import from arrg.a2a instead:
    from arrg.a2a import Message, MessageRole, Task, TaskState
"""

import warnings

# Re-export A2A types for backward compatibility
from arrg.a2a import (
    Message,
    MessageRole,
    Task,
    TaskState,
    TextPart,
    DataPart,
)

# Deprecated aliases
A2AMessage = Message
MessageType = TaskState

warnings.warn(
    "arrg.protocol.message is deprecated. Use arrg.a2a instead.",
    DeprecationWarning,
    stacklevel=2,
)
