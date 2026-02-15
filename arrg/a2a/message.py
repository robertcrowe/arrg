"""
A2A Protocol - Message and Part types.

Implements Message and Part (TextPart, FilePart, DataPart) per A2A Protocol v1.0.

Per the spec, Messages are the communication units within Tasks. Each Message
has a role (user or agent) and contains one or more Parts. Parts carry the
actual content and come in three types:
- TextPart: Plain text or markdown content
- FilePart: File content (inline bytes or URI reference)
- DataPart: Structured JSON data

See: https://github.com/google/A2A/blob/main/specification/a2a.proto
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
import json
import uuid


class MessageRole(Enum):
    """
    Role of the message sender.
    
    Per A2A spec:
    - USER: Message from the requesting agent/client
    - AGENT: Message from the responding agent
    """
    USER = "user"
    AGENT = "agent"


@dataclass
class TextPart:
    """
    Text content part.
    
    Per A2A spec: Contains plain text or markdown content.
    """
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result: Dict[str, Any] = {"type": "text", "text": self.text}
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextPart":
        """Deserialize from dictionary."""
        return cls(text=data["text"], metadata=data.get("metadata", {}))


@dataclass
class FilePart:
    """
    File content part.
    
    Per A2A spec: Contains file data either inline (as bytes/base64)
    or as a URI reference. Includes MIME type for content negotiation.
    """
    name: str = ""
    mime_type: str = "application/octet-stream"
    uri: Optional[str] = None
    data: Optional[str] = None  # base64-encoded inline data
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result: Dict[str, Any] = {"type": "file", "mimeType": self.mime_type}
        if self.name:
            result["name"] = self.name
        if self.uri:
            result["uri"] = self.uri
        if self.data:
            result["data"] = self.data
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilePart":
        """Deserialize from dictionary."""
        return cls(
            name=data.get("name", ""),
            mime_type=data.get("mimeType", "application/octet-stream"),
            uri=data.get("uri"),
            data=data.get("data"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DataPart:
    """
    Structured data content part.
    
    Per A2A spec: Contains structured JSON data for machine-readable content.
    """
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result: Dict[str, Any] = {"type": "data", "data": self.data}
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataPart":
        """Deserialize from dictionary."""
        return cls(data=data.get("data", {}), metadata=data.get("metadata", {}))


# Union type for all Part variants
Part = Union[TextPart, FilePart, DataPart]


def part_from_dict(data: Dict[str, Any]) -> Part:
    """Deserialize a Part from dictionary based on its type field."""
    part_type = data.get("type", "text")
    if part_type == "text":
        return TextPart.from_dict(data)
    elif part_type == "file":
        return FilePart.from_dict(data)
    elif part_type == "data":
        return DataPart.from_dict(data)
    else:
        # Default to TextPart for unknown types
        return TextPart(text=str(data))


@dataclass
class Message:
    """
    A2A Protocol Message - Communication unit within a Task.
    
    Per the spec: Messages carry the actual content exchanged between agents
    during task execution. Each message has a role (user/agent) and contains
    one or more Parts (text, file, or structured data).
    
    Messages are accumulated in the Task's history as the conversation
    progresses through the task lifecycle.
    """
    role: MessageRole
    parts: List[Part] = field(default_factory=list)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Extension fields for ARRG internal routing (not part of A2A spec)
    sender: str = ""
    task_id: str = ""
    in_reply_to: Optional[str] = None

    def get_text(self) -> str:
        """Extract all text content from the message parts."""
        texts = []
        for part in self.parts:
            if isinstance(part, TextPart):
                texts.append(part.text)
        return "\n".join(texts)

    def get_data(self) -> Optional[Dict[str, Any]]:
        """Extract the first DataPart's data, if any."""
        for part in self.parts:
            if isinstance(part, DataPart):
                return part.data
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Message to dictionary."""
        result: Dict[str, Any] = {
            "role": self.role.value,
            "parts": [part.to_dict() for part in self.parts],
            "messageId": self.message_id,
            "timestamp": self.timestamp,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        if self.sender:
            result["metadata"] = result.get("metadata", {})
            result["metadata"]["sender"] = self.sender
        if self.task_id:
            result["metadata"] = result.get("metadata", {})
            result["metadata"]["taskId"] = self.task_id
        if self.in_reply_to:
            result["metadata"] = result.get("metadata", {})
            result["metadata"]["inReplyTo"] = self.in_reply_to
        return result

    def to_json(self) -> str:
        """Serialize Message to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Deserialize Message from dictionary."""
        parts = [part_from_dict(p) for p in data.get("parts", [])]
        metadata = data.get("metadata", {})
        return cls(
            role=MessageRole(data["role"]),
            parts=parts,
            message_id=data.get("messageId", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata={k: v for k, v in metadata.items() if k not in ("sender", "taskId", "inReplyTo")},
            sender=metadata.get("sender", ""),
            task_id=metadata.get("taskId", ""),
            in_reply_to=metadata.get("inReplyTo"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Deserialize Message from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @staticmethod
    def create_user_message(
        text: str = "",
        data: Optional[Dict[str, Any]] = None,
        sender: str = "",
        task_id: str = "",
        **kwargs,
    ) -> "Message":
        """
        Factory: Create a user-role message.
        
        Args:
            text: Text content (creates TextPart)
            data: Structured data (creates DataPart)
            sender: Sender identifier
            task_id: Associated task ID
        """
        parts: List[Part] = []
        if text:
            parts.append(TextPart(text=text))
        if data:
            parts.append(DataPart(data=data))
        return Message(
            role=MessageRole.USER,
            parts=parts,
            sender=sender,
            task_id=task_id,
            **kwargs,
        )

    @staticmethod
    def create_agent_message(
        text: str = "",
        data: Optional[Dict[str, Any]] = None,
        sender: str = "",
        task_id: str = "",
        in_reply_to: Optional[str] = None,
        **kwargs,
    ) -> "Message":
        """
        Factory: Create an agent-role message.
        
        Args:
            text: Text content (creates TextPart)
            data: Structured data (creates DataPart)
            sender: Sender identifier
            task_id: Associated task ID
            in_reply_to: ID of the message being replied to
        """
        parts: List[Part] = []
        if text:
            parts.append(TextPart(text=text))
        if data:
            parts.append(DataPart(data=data))
        return Message(
            role=MessageRole.AGENT,
            parts=parts,
            sender=sender,
            task_id=task_id,
            in_reply_to=in_reply_to,
            **kwargs,
        )
