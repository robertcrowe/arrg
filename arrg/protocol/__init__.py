"""Protocol module initialization."""

from .message import A2AMessage, MessageType, TaskStatus
from .workspace import SharedWorkspace

__all__ = [
    'A2AMessage',
    'MessageType', 
    'TaskStatus',
    'SharedWorkspace',
]
