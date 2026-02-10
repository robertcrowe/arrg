"""ARRG - Automated Research Report Generator

A multi-agent system for generating comprehensive research reports.
"""

from arrg.core import Orchestrator
from arrg.protocol import A2AMessage, MessageType, TaskStatus, SharedWorkspace
from arrg.agents import (
    BaseAgent,
    PlanningAgent,
    ResearchAgent,
    AnalysisAgent,
    WritingAgent,
    QAAgent,
)

__version__ = "0.1.0"

__all__ = [
    'Orchestrator',
    'A2AMessage',
    'MessageType',
    'TaskStatus',
    'SharedWorkspace',
    'BaseAgent',
    'PlanningAgent',
    'ResearchAgent',
    'AnalysisAgent',
    'WritingAgent',
    'QAAgent',
]
