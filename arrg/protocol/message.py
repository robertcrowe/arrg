"""
A2A Message types and structures for agent communication.

This module provides backward-compatible A2A messaging while supporting
the new A2A v1.0 protocol structures.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime
import uuid


class MessageType(str, Enum):
    """
    Types of A2A messages.
    
    Maintains backward compatibility while supporting A2A v1.0 message types.
    """
    # Legacy message types (backward compatibility)
    PROPOSAL = "proposal"
    TASK_REQUEST = "task_request"
    TASK_COMPLETE = "task_complete"
    TASK_REJECTED = "task_rejected"
    DATA_TRANSFER = "data_transfer"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"
    ERROR = "error"
    
    # A2A v1.0 message types
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    STREAM = "stream"


class TaskStatus(str, Enum):
    """
    Status of a task.
    
    Aligned with A2A v1.0 Task status values.
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    COMPLETED = "completed"  # A2A v1.0 variant
    REJECTED = "rejected"
    FAILED = "failed"  # A2A v1.0
    CANCELED = "canceled"  # A2A v1.0
    ERROR = "error"


@dataclass
class A2AMessage:
    """
    Standardized A2A Protocol message structure.
    
    This implementation supports both the legacy ARRG message format
    and the A2A v1.0 protocol specification.
    
    Attributes:
        message_type: Type of message being sent
        sender: ID of the sending agent
        receiver: ID of the receiving agent
        payload: Message content and data
        message_id: Unique identifier for this message
        timestamp: When the message was created
        in_reply_to: ID of message being replied to (optional)
        metadata: Additional context or routing information
        task_id: Associated task ID (A2A v1.0)
        content: Message content string (A2A v1.0)
    """
    message_type: MessageType
    sender: str
    receiver: str
    payload: Dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    in_reply_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None
    content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        result = {
            "message_type": self.message_type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "payload": self.payload,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "in_reply_to": self.in_reply_to,
            "metadata": self.metadata,
        }
        
        if self.task_id:
            result["task_id"] = self.task_id
        if self.content:
            result["content"] = self.content
            
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        """Create message from dictionary."""
        return cls(
            message_type=MessageType(data["message_type"]),
            sender=data["sender"],
            receiver=data["receiver"],
            payload=data["payload"],
            message_id=data.get("message_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            in_reply_to=data.get("in_reply_to"),
            metadata=data.get("metadata", {}),
            task_id=data.get("task_id"),
            content=data.get("content"),
        )

    def create_reply(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None,
    ) -> "A2AMessage":
        """Create a reply message to this message."""
        return A2AMessage(
            message_type=message_type,
            sender=self.receiver,  # Original receiver becomes sender
            receiver=self.sender,  # Original sender becomes receiver
            payload=payload,
            in_reply_to=self.message_id,
            metadata=metadata or {},
            task_id=self.task_id,
            content=content,
        )
    
    def to_a2a_v1(self) -> Dict[str, Any]:
        """Convert to A2A v1.0 message format."""
        from ..a2a import A2AMessage as A2AV1Message, MessageType as A2AV1MessageType, MessageRole
        
        # Map legacy message types to A2A v1.0
        type_mapping = {
            MessageType.TASK_REQUEST: A2AV1MessageType.REQUEST,
            MessageType.TASK_COMPLETE: A2AV1MessageType.RESPONSE,
            MessageType.TASK_REJECTED: A2AV1MessageType.ERROR,
            MessageType.DATA_TRANSFER: A2AV1MessageType.RESPONSE,
            MessageType.CAPABILITY_QUERY: A2AV1MessageType.REQUEST,
            MessageType.CAPABILITY_RESPONSE: A2AV1MessageType.RESPONSE,
            MessageType.ERROR: A2AV1MessageType.ERROR,
            MessageType.REQUEST: A2AV1MessageType.REQUEST,
            MessageType.RESPONSE: A2AV1MessageType.RESPONSE,
            MessageType.NOTIFICATION: A2AV1MessageType.NOTIFICATION,
        }
        
        a2a_type = type_mapping.get(self.message_type, A2AV1MessageType.REQUEST)
        
        v1_msg = A2AV1Message(
            message_id=self.message_id,
            message_type=a2a_type,
            role=MessageRole.USER if self.message_type == MessageType.REQUEST else MessageRole.ASSISTANT,
            content=self.content or str(self.payload),
            task_id=self.task_id,
            sender_id=self.sender,
            receiver_id=self.receiver,
            timestamp=self.timestamp,
            parent_message_id=self.in_reply_to,
            metadata=self.metadata,
        )
        
        return v1_msg.to_dict()
