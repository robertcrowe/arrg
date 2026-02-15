"""
A2A Protocol - Artifact.

Implements the Artifact data structure per the A2A Protocol v1.0.

Per the spec, Artifacts are outputs produced by agents during task execution.
They contain one or more Parts (text, file, or data) and are accumulated
in the Task's artifacts list as the agent produces output.

Artifacts are distinct from Messages: Messages are the conversation between
agents, while Artifacts are the deliverable outputs of the task.

See: https://github.com/google/A2A/blob/main/specification/a2a.proto
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
import json
import uuid


# Import Part types from message module
from .message import Part, TextPart, FilePart, DataPart, part_from_dict


@dataclass
class Artifact:
    """
    A2A Protocol Artifact - Output produced during task execution.
    
    Per the spec: Artifacts represent the deliverable outputs of a task.
    Each artifact has a unique ID, a name, and contains one or more Parts.
    Artifacts are accumulated in the Task as the agent produces output.
    
    In ARRG, artifacts represent research plans, research data, analysis
    results, written reports, and QA reviews.
    """
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    parts: List[Part] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def get_text(self) -> str:
        """Extract all text content from artifact parts."""
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
        """Serialize Artifact to dictionary."""
        result: Dict[str, Any] = {
            "artifactId": self.artifact_id,
            "parts": [part.to_dict() for part in self.parts],
        }
        if self.name:
            result["name"] = self.name
        if self.description:
            result["description"] = self.description
        if self.metadata:
            result["metadata"] = self.metadata
        result["createdAt"] = self.created_at
        return result

    def to_json(self) -> str:
        """Serialize Artifact to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """Deserialize Artifact from dictionary."""
        parts = [part_from_dict(p) for p in data.get("parts", [])]
        return cls(
            artifact_id=data.get("artifactId", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            parts=parts,
            metadata=data.get("metadata", {}),
            created_at=data.get("createdAt", datetime.now(timezone.utc).isoformat()),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Artifact":
        """Deserialize Artifact from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @staticmethod
    def create_text_artifact(
        text: str,
        name: str = "",
        description: str = "",
        **kwargs,
    ) -> "Artifact":
        """Factory: Create an artifact with text content."""
        return Artifact(
            name=name,
            description=description,
            parts=[TextPart(text=text)],
            **kwargs,
        )

    @staticmethod
    def create_data_artifact(
        data: Dict[str, Any],
        name: str = "",
        description: str = "",
        **kwargs,
    ) -> "Artifact":
        """Factory: Create an artifact with structured data."""
        return Artifact(
            name=name,
            description=description,
            parts=[DataPart(data=data)],
            **kwargs,
        )
