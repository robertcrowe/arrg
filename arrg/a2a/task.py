"""
A2A Protocol - Task and TaskState.

Implements the Task data structure and TaskState enum per the A2A Protocol v1.0.

Per the spec, a Task is the fundamental unit of work in A2A. It has:
- A unique ID and optional context_id for grouping related tasks
- A status containing the current TaskState and optional message
- History of messages exchanged during task execution
- Artifacts produced as output
- Metadata for extensibility

TaskState follows the A2A state machine:
  submitted → working → {completed, failed, canceled, input_required, rejected, auth_required}

See: https://github.com/google/A2A/blob/main/specification/a2a.proto
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import json
import uuid


class TaskState(Enum):
    """
    A2A Protocol Task States.
    
    Per the spec protobuf TaskState enum:
    - SUBMITTED: Task has been received but not yet started
    - WORKING: Task is actively being processed
    - COMPLETED: Task finished successfully
    - FAILED: Task encountered an error
    - CANCELED: Task was canceled by request
    - INPUT_REQUIRED: Agent needs additional input from the caller
    - REJECTED: Agent rejected the task
    - AUTH_REQUIRED: Authentication is needed to proceed
    """
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    INPUT_REQUIRED = "input_required"
    REJECTED = "rejected"
    AUTH_REQUIRED = "auth_required"


@dataclass
class TaskStatus:
    """
    Current status of a Task.
    
    Per A2A spec: Contains the state enum and an optional message
    providing additional context about the current state.
    """
    state: TaskState
    message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result: Dict[str, Any] = {"state": self.state.value}
        if self.message:
            result["message"] = self.message
        result["timestamp"] = self.timestamp
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskStatus":
        """Deserialize from dictionary."""
        return cls(
            state=TaskState(data["state"]),
            message=data.get("message"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )

    @property
    def is_terminal(self) -> bool:
        """Check if this status represents a terminal state."""
        return self.state in (
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELED,
            TaskState.REJECTED,
        )


@dataclass
class Task:
    """
    A2A Protocol Task - Unit of work between agents.
    
    Per the spec: A Task represents a unit of work sent from one agent
    to another. It progresses through states (submitted → working → 
    completed/failed/etc.) and accumulates messages and artifacts.
    
    The Task is identified by a unique id and optionally grouped with
    related tasks via context_id.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_id: Optional[str] = None
    status: TaskStatus = field(default_factory=lambda: TaskStatus(state=TaskState.SUBMITTED))
    history: List[Any] = field(default_factory=list)  # List of Message objects
    artifacts: List[Any] = field(default_factory=list)  # List of Artifact objects
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_state(self, state: TaskState, message: Optional[str] = None) -> None:
        """
        Transition task to a new state.
        
        Args:
            state: New TaskState
            message: Optional status message
        """
        self.status = TaskStatus(state=state, message=message)

    def add_to_history(self, message: Any) -> None:
        """Add a message to the task history."""
        self.history.append(message)

    def add_artifact(self, artifact: Any) -> None:
        """Add an artifact to the task outputs."""
        self.artifacts.append(artifact)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Task to dictionary."""
        result: Dict[str, Any] = {
            "id": self.id,
            "status": self.status.to_dict(),
        }
        if self.context_id:
            result["contextId"] = self.context_id
        if self.history:
            result["history"] = [
                msg.to_dict() if hasattr(msg, "to_dict") else msg
                for msg in self.history
            ]
        if self.artifacts:
            result["artifacts"] = [
                art.to_dict() if hasattr(art, "to_dict") else art
                for art in self.artifacts
            ]
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def to_json(self) -> str:
        """Serialize Task to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Deserialize Task from dictionary."""
        status = TaskStatus.from_dict(data["status"]) if "status" in data else TaskStatus(state=TaskState.SUBMITTED)
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            context_id=data.get("contextId"),
            status=status,
            history=data.get("history", []),
            artifacts=data.get("artifacts", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Task":
        """Deserialize Task from JSON string."""
        return cls.from_dict(json.loads(json_str))
