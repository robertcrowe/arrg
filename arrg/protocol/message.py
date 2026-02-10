"""A2A Message types and structures for agent communication."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime


class MessageType(str, Enum):
    """Types of A2A messages."""
    PROPOSAL = "proposal"
    TASK_REQUEST = "task_request"
    TASK_COMPLETE = "task_complete"
    TASK_REJECTED = "task_rejected"
    DATA_TRANSFER = "data_transfer"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"
    ERROR = "error"


class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class A2AMessage:
    """
    Standardized A2A Protocol message structure.
    
    Attributes:
        message_type: Type of message being sent
        sender: ID of the sending agent
        receiver: ID of the receiving agent
        payload: Message content and data
        message_id: Unique identifier for this message
        timestamp: When the message was created
        in_reply_to: ID of message being replied to (optional)
        metadata: Additional context or routing information
    """
    message_type: MessageType
    sender: str
    receiver: str
    payload: Dict[str, Any]
    message_id: str = field(default_factory=lambda: datetime.now().isoformat())
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    in_reply_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "message_type": self.message_type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "payload": self.payload,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "in_reply_to": self.in_reply_to,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        """Create message from dictionary."""
        return cls(
            message_type=MessageType(data["message_type"]),
            sender=data["sender"],
            receiver=data["receiver"],
            payload=data["payload"],
            message_id=data.get("message_id", datetime.now().isoformat()),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            in_reply_to=data.get("in_reply_to"),
            metadata=data.get("metadata", {}),
        )

    def create_reply(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "A2AMessage":
        """Create a reply message to this message."""
        return A2AMessage(
            message_type=message_type,
            sender=self.receiver,  # Original receiver becomes sender
            receiver=self.sender,  # Original sender becomes receiver
            payload=payload,
            in_reply_to=self.message_id,
            metadata=metadata or {},
        )
