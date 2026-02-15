"""
A2A Protocol - AgentCard and related types.

Implements the AgentCard, AgentSkill, AgentProvider, and AgentCapabilities
data structures per the A2A Protocol v1.0 specification.

AgentCards are the primary mechanism for agent discovery and capability
advertisement. They are served at /.well-known/agent.json for open discovery.

See: https://github.com/google/A2A/blob/main/specification/a2a.proto
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json


@dataclass
class AgentProvider:
    """
    Provider information for an agent.
    
    Per A2A spec: Contains the organization and URL of the agent provider.
    """
    organization: str
    url: str = ""


@dataclass
class AgentCapabilities:
    """
    Capabilities supported by an agent.
    
    Per A2A spec: Declares what protocol features the agent supports.
    """
    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = True


@dataclass
class AgentSkill:
    """
    A skill that an agent can perform.
    
    Per A2A spec: Skills describe specific capabilities with input/output
    modes, tags for discovery, and example prompts.
    """
    id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    input_modes: List[str] = field(default_factory=lambda: ["text/plain", "application/json"])
    output_modes: List[str] = field(default_factory=lambda: ["text/plain", "application/json"])


@dataclass
class AgentCard:
    """
    Agent Card - Identity and capability advertisement.
    
    Per A2A spec: The AgentCard is the primary mechanism for agent discovery.
    It contains the agent's identity, capabilities, skills, and connection
    information. Served at /.well-known/agent.json for open discovery.
    
    Fields align with the A2A Protocol protobuf definition:
    - name, description, version: Agent identity
    - url: Agent's A2A endpoint URL
    - provider: Organization operating the agent
    - capabilities: Protocol features supported
    - skills: Specific tasks the agent can perform
    - default_input_modes/default_output_modes: Supported content types
    - security_schemes: Authentication mechanisms
    """
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    provider: Optional[AgentProvider] = None
    capabilities: Optional[AgentCapabilities] = None
    skills: List[AgentSkill] = field(default_factory=list)
    default_input_modes: List[str] = field(default_factory=lambda: ["text/plain", "application/json"])
    default_output_modes: List[str] = field(default_factory=lambda: ["text/plain", "application/json"])
    security_schemes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize AgentCard to dictionary (JSON-compatible for /.well-known/agent.json)."""
        result = {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
        }
        if self.provider:
            result["provider"] = {
                "organization": self.provider.organization,
                "url": self.provider.url,
            }
        if self.capabilities:
            result["capabilities"] = {
                "streaming": self.capabilities.streaming,
                "pushNotifications": self.capabilities.push_notifications,
                "stateTransitionHistory": self.capabilities.state_transition_history,
            }
        if self.skills:
            result["skills"] = [
                {
                    "id": skill.id,
                    "name": skill.name,
                    "description": skill.description,
                    "tags": skill.tags,
                    "examples": skill.examples,
                    "inputModes": skill.input_modes,
                    "outputModes": skill.output_modes,
                }
                for skill in self.skills
            ]
        if self.security_schemes:
            result["securitySchemes"] = self.security_schemes
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def to_json(self) -> str:
        """Serialize AgentCard to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        """Deserialize AgentCard from dictionary."""
        provider = None
        if "provider" in data:
            provider = AgentProvider(
                organization=data["provider"].get("organization", ""),
                url=data["provider"].get("url", ""),
            )
        
        capabilities = None
        if "capabilities" in data:
            cap_data = data["capabilities"]
            capabilities = AgentCapabilities(
                streaming=cap_data.get("streaming", False),
                push_notifications=cap_data.get("pushNotifications", False),
                state_transition_history=cap_data.get("stateTransitionHistory", True),
            )
        
        skills = []
        for skill_data in data.get("skills", []):
            skills.append(AgentSkill(
                id=skill_data["id"],
                name=skill_data["name"],
                description=skill_data["description"],
                tags=skill_data.get("tags", []),
                examples=skill_data.get("examples", []),
                input_modes=skill_data.get("inputModes", ["text/plain"]),
                output_modes=skill_data.get("outputModes", ["text/plain"]),
            ))
        
        return cls(
            name=data["name"],
            description=data["description"],
            url=data.get("url", ""),
            version=data.get("version", "1.0.0"),
            provider=provider,
            capabilities=capabilities,
            skills=skills,
            default_input_modes=data.get("defaultInputModes", ["text/plain", "application/json"]),
            default_output_modes=data.get("defaultOutputModes", ["text/plain", "application/json"]),
            security_schemes=data.get("securitySchemes", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AgentCard":
        """Deserialize AgentCard from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def has_skill(self, skill_id: str) -> bool:
        """Check if agent has a specific skill."""
        return any(s.id == skill_id for s in self.skills)

    def get_skill(self, skill_id: str) -> Optional[AgentSkill]:
        """Get a specific skill by ID."""
        for skill in self.skills:
            if skill.id == skill_id:
                return skill
        return None

    def supports_input_mode(self, mode: str) -> bool:
        """Check if agent supports a specific input content type."""
        return mode in self.default_input_modes

    def supports_output_mode(self, mode: str) -> bool:
        """Check if agent supports a specific output content type."""
        return mode in self.default_output_modes
