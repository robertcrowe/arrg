"""
Orchestrator for managing the multi-agent workflow.

All agent-to-agent communication uses the A2A Protocol v1.0:
- Tasks with lifecycle states (submitted → working → completed/failed)
- Messages with typed Parts (TextPart, DataPart)
- Artifacts for deliverable outputs
- AgentCards for capability advertisement

Tool-calling uses MCP 2025-11-25 (complementary to A2A).
"""

from typing import Any, Callable, Dict, Optional
import logging
from pathlib import Path

from arrg.agents import (
    PlanningAgent,
    ResearchAgent,
    AnalysisAgent,
    WritingAgent,
    QAAgent,
)
from arrg.a2a import (
    Task,
    TaskState,
    TaskStatus,
    Message,
    MessageRole,
    TextPart,
    DataPart,
    Artifact,
)
from arrg.protocol import SharedWorkspace


class Orchestrator:
    """
    Orchestrates the multi-agent research report generation workflow.

    Uses A2A Protocol v1.0 for all agent communication:
    - Creates Tasks for each workflow phase
    - Sends Messages with typed Parts to agents
    - Receives completed Tasks with Artifacts
    - Tracks task lifecycle states per the A2A state machine
    """

    def __init__(
        self,
        api_key: str,
        provider_endpoint: str = "Tetrate",
        models: Optional[Dict[str, str]] = None,
        workspace_dir: Optional[Path] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            api_key: API key for the model provider
            provider_endpoint: API provider endpoint
            models: Dict mapping agent names to model strings
            workspace_dir: Directory for shared workspace
            stream_callback: Optional callback for streaming output
        """
        self.api_key = api_key
        self.provider_endpoint = provider_endpoint
        self.stream_callback = stream_callback
        self.logger = logging.getLogger("arrg.orchestrator")

        # Set default models if not provided
        default_model = "claude-haiku-4-5"
        default_models = {
            "planning": default_model,
            "research": default_model,
            "analysis": default_model,
            "writing": default_model,
            "qa": default_model,
        }

        # Merge provided models with defaults
        if models:
            self.models = {**default_models, **models}
        else:
            self.models = default_models

        # Initialize shared workspace for artifact storage
        self.workspace = SharedWorkspace(workspace_dir)

        # Initialize agents with per-agent models
        self.agents = {
            "planning": PlanningAgent(
                agent_id="planning",
                model=self.models["planning"],
                workspace=self.workspace,
                api_key=api_key,
                provider_endpoint=provider_endpoint,
                stream_callback=stream_callback,
            ),
            "research": ResearchAgent(
                agent_id="research",
                model=self.models["research"],
                workspace=self.workspace,
                api_key=api_key,
                provider_endpoint=provider_endpoint,
                stream_callback=stream_callback,
            ),
            "analysis": AnalysisAgent(
                agent_id="analysis",
                model=self.models["analysis"],
                workspace=self.workspace,
                api_key=api_key,
                provider_endpoint=provider_endpoint,
                stream_callback=stream_callback,
            ),
            "writing": WritingAgent(
                agent_id="writing",
                model=self.models["writing"],
                workspace=self.workspace,
                api_key=api_key,
                provider_endpoint=provider_endpoint,
                stream_callback=stream_callback,
            ),
            "qa": QAAgent(
                agent_id="qa",
                model=self.models["qa"],
                workspace=self.workspace,
                api_key=api_key,
                provider_endpoint=provider_endpoint,
                stream_callback=stream_callback,
            ),
        }

        # Workflow progress tracking using A2A TaskState
        self.current_state = TaskState.SUBMITTED
        self.workflow_progress: Dict[str, str] = {
            "planning": TaskState.SUBMITTED.value,
            "research": TaskState.SUBMITTED.value,
            "analysis": TaskState.SUBMITTED.value,
            "writing": TaskState.SUBMITTED.value,
            "qa": TaskState.SUBMITTED.value,
        }

        # A2A message and task history
        self.message_history: list[Message] = []
        self.tasks: Dict[str, Task] = {}

        # QA revision tracking
        self.max_qa_retries = 2
        self.qa_retry_count = 0

    def generate_report(
        self, topic: str, requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a research report on the given topic.

        Executes the 5-phase A2A workflow:
        Planning → Research → Analysis → Writing → QA (with revision loop)

        Args:
            topic: Research topic
            requirements: Optional requirements and constraints

        Returns:
            Final report and metadata
        """
        self.logger.info(f"Starting report generation for topic: {topic}")
        self.stream_output(f"Starting report generation for: {topic}")
        self.current_state = TaskState.WORKING

        try:
            # Step 1: Planning
            self.stream_output("\n=== PHASE 1: PLANNING ===")
            self.workflow_progress["planning"] = TaskState.WORKING.value
            plan_result = self._execute_planning(topic, requirements or {})
            self.workflow_progress["planning"] = TaskState.COMPLETED.value

            # Step 2: Research
            self.stream_output("\n=== PHASE 2: RESEARCH ===")
            self.workflow_progress["research"] = TaskState.WORKING.value
            research_result = self._execute_research(plan_result)
            self.workflow_progress["research"] = TaskState.COMPLETED.value

            # Step 3: Analysis
            self.stream_output("\n=== PHASE 3: ANALYSIS ===")
            self.workflow_progress["analysis"] = TaskState.WORKING.value
            analysis_result = self._execute_analysis(research_result, plan_result)
            self.workflow_progress["analysis"] = TaskState.COMPLETED.value

            # Step 4: Writing
            self.stream_output("\n=== PHASE 4: WRITING ===")
            self.workflow_progress["writing"] = TaskState.WORKING.value
            writing_result = self._execute_writing(analysis_result, plan_result)
            self.workflow_progress["writing"] = TaskState.COMPLETED.value

            # Step 5: QA with revision loop
            self.stream_output("\n=== PHASE 5: QUALITY ASSURANCE ===")
            self.workflow_progress["qa"] = TaskState.WORKING.value
            self.qa_retry_count = 0
            qa_result = None

            while self.qa_retry_count <= self.max_qa_retries:
                qa_result = self._execute_qa(writing_result)

                # Check if QA approved the report
                if qa_result.get("approved", False):
                    self.stream_output("✓ QA Review: Report APPROVED")
                    self.workflow_progress["qa"] = TaskState.COMPLETED.value
                    break
                else:
                    self.qa_retry_count += 1
                    if self.qa_retry_count <= self.max_qa_retries:
                        self.stream_output(
                            f"✗ QA Review: Report REJECTED (Attempt {self.qa_retry_count}/{self.max_qa_retries})"
                        )
                        self.stream_output("→ Sending back to Writing Agent for revision...")

                        # Send rejection back to Writing Agent with QA feedback
                        self.workflow_progress["writing"] = TaskState.WORKING.value
                        writing_result = self._execute_writing_revision(
                            analysis_result, plan_result, qa_result
                        )
                        self.workflow_progress["writing"] = TaskState.COMPLETED.value
                    else:
                        self.stream_output(
                            f"✗ QA Review: Report REJECTED (Max retries reached)"
                        )
                        self.stream_output("→ Proceeding with current version despite issues...")
                        self.workflow_progress["qa"] = TaskState.COMPLETED.value
                        break

            self.current_state = TaskState.COMPLETED
            self.stream_output("\n=== REPORT GENERATION COMPLETE ===")

            # Compile final results
            report = self.workspace.retrieve(writing_result["report_reference"])
            qa_report = self.workspace.retrieve(qa_result["qa_reference"])

            return {
                "status": "success",
                "report": report,
                "qa_results": qa_report,
                "report_reference": writing_result["report_reference"],
                "qa_reference": qa_result["qa_reference"],
                "metadata": {
                    "topic": topic,
                    "workflow_status": self.workflow_progress,
                },
            }

        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            self.stream_output(f"\n!!! ERROR: {str(e)} !!!")
            self.current_state = TaskState.FAILED
            return {
                "status": "error",
                "error": str(e),
                "workflow_status": self.workflow_progress,
            }

    def _send_task_to_agent(
        self,
        agent_name: str,
        task_description: str,
        payload: Dict[str, Any],
        context_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send an A2A Task to an agent and return the result.

        Per A2A Protocol: Creates a Task in SUBMITTED state, sends a user
        Message with the payload as a DataPart, and calls the agent's
        process_task method. Returns the result data from the completed task.

        Args:
            agent_name: Name of the target agent
            task_description: Human-readable task description
            payload: Data to send in the Message's DataPart
            context_id: Optional context ID for grouping related tasks

        Returns:
            Result data from the completed task's artifacts

        Raises:
            RuntimeError: If the task fails
        """
        # Create A2A Task
        task = Task(context_id=context_id)
        task.metadata["description"] = task_description
        task.metadata["agent"] = agent_name
        self.tasks[task.id] = task

        # Create A2A Message with typed Parts
        message = Message.create_user_message(
            text=task_description,
            data=payload,
            sender="orchestrator",
            task_id=task.id,
        )

        self.message_history.append(message)
        self.logger.info(f"Created A2A Task {task.id} for {agent_name}: {task_description}")

        try:
            # Send task to agent via A2A process_task
            completed_task = self.agents[agent_name].process_task(task, message)
            self.tasks[task.id] = completed_task

            # Record response messages from task history
            for msg in completed_task.history:
                if isinstance(msg, Message) and msg.role == MessageRole.AGENT:
                    self.message_history.append(msg)

            # Check task state
            if completed_task.status.state == TaskState.FAILED:
                error = completed_task.status.message or "Unknown error"
                raise RuntimeError(f"{agent_name} failed: {error}")

            # Extract result data from the task's artifacts
            result_data = {}
            for artifact in completed_task.artifacts:
                if isinstance(artifact, Artifact):
                    artifact_data = artifact.get_data()
                    if artifact_data:
                        result_data.update(artifact_data)

            return result_data

        except Exception as e:
            task.update_state(TaskState.FAILED, message=str(e))
            raise

    def _execute_planning(
        self, topic: str, requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the planning phase using A2A Task."""
        return self._send_task_to_agent(
            agent_name="planning",
            task_description=f"Create research plan for topic: {topic}",
            payload={
                "topic": topic,
                "requirements": requirements,
            },
        )

    def _execute_research(self, plan_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research phase using A2A Task."""
        return self._send_task_to_agent(
            agent_name="research",
            task_description="Conduct research on questions from plan",
            payload={
                "research_questions": plan_result["research_questions"],
                "plan_reference": plan_result["plan_reference"],
            },
        )

    def _execute_analysis(
        self, research_result: Dict[str, Any], plan_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the analysis phase using A2A Task."""
        return self._send_task_to_agent(
            agent_name="analysis",
            task_description="Analyze research data and synthesize insights",
            payload={
                "data_reference": research_result["data_reference"],
                "plan_reference": plan_result["plan_reference"],
            },
        )

    def _execute_writing(
        self, analysis_result: Dict[str, Any], plan_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the writing phase using A2A Task."""
        return self._send_task_to_agent(
            agent_name="writing",
            task_description="Write comprehensive research report",
            payload={
                "analysis_reference": analysis_result["analysis_reference"],
                "plan_reference": plan_result["plan_reference"],
            },
        )

    def _execute_qa(self, writing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the QA phase using A2A Task."""
        return self._send_task_to_agent(
            agent_name="qa",
            task_description="Quality assurance review of report",
            payload={
                "report_reference": writing_result["report_reference"],
            },
        )

    def _execute_writing_revision(
        self,
        analysis_result: Dict[str, Any],
        plan_result: Dict[str, Any],
        qa_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the writing revision phase using A2A Task with QA feedback."""
        return self._send_task_to_agent(
            agent_name="writing",
            task_description="Revise report based on QA feedback",
            payload={
                "analysis_reference": analysis_result["analysis_reference"],
                "plan_reference": plan_result["plan_reference"],
                "report_reference": qa_result.get("qa_reference", ""),
                "qa_feedback": qa_result,
                "revision_required": True,
            },
        )

    def stream_output(self, text: str):
        """Stream output to the dashboard console."""
        if self.stream_callback:
            self.stream_callback(text)
        self.logger.info(text)

    def get_progress(self) -> Dict[str, Any]:
        """Get current workflow progress using A2A TaskState."""
        return {
            "status": self.current_state.value,
            "phases": self.workflow_progress,
        }

    def get_message_log(self) -> str:
        """
        Get a downloadable log of all A2A Protocol messages and agent outputs.

        Returns:
            Formatted log string
        """
        log_lines = [
            "=" * 80,
            "ARRG WORKFLOW LOG - A2A Protocol v1.0",
            "=" * 80,
            "",
            f"Models Configuration:",
        ]

        for agent, model in self.models.items():
            log_lines.append(f"  {agent}: {model}")

        log_lines.extend([
            "",
            "=" * 80,
            "A2A TASK HISTORY",
            "=" * 80,
            "",
        ])

        for task_id, task in self.tasks.items():
            log_lines.extend([
                f"Task: {task_id}",
                f"  State: {task.status.state.value}",
                f"  Description: {task.metadata.get('description', 'N/A')}",
                f"  Agent: {task.metadata.get('agent', 'N/A')}",
                f"  Artifacts: {len(task.artifacts)}",
                f"  History Messages: {len(task.history)}",
                "",
            ])

        log_lines.extend([
            "=" * 80,
            "A2A MESSAGE HISTORY",
            "=" * 80,
            "",
        ])

        for i, msg in enumerate(self.message_history, 1):
            text_preview = msg.get_text()[:200] if msg.get_text() else "N/A"
            data_preview = str(msg.get_data())[:200] if msg.get_data() else "N/A"
            log_lines.extend([
                f"Message {i}:",
                f"  ID: {msg.message_id}",
                f"  Role: {msg.role.value}",
                f"  Sender: {msg.sender}",
                f"  Task ID: {msg.task_id}",
                f"  Timestamp: {msg.timestamp}",
                f"  In Reply To: {msg.in_reply_to or 'N/A'}",
                f"  Text: {text_preview}",
                f"  Data: {data_preview}",
                "",
            ])

        log_lines.extend([
            "=" * 80,
            "AGENT MESSAGE HISTORIES",
            "=" * 80,
            "",
        ])

        for agent_name, agent in self.agents.items():
            log_lines.extend([
                f"Agent: {agent_name} (Model: {agent.model})",
                f"  AgentCard: {agent.agent_card.name}",
                f"  Skills: {[s.id for s in agent.agent_card.skills]}",
                "-" * 40,
            ])

            for i, msg in enumerate(agent.message_history, 1):
                log_lines.extend([
                    f"  [{i}] {msg.role.value}: {msg.sender}",
                    f"      ID: {msg.message_id}",
                    f"      Timestamp: {msg.timestamp}",
                ])

            log_lines.append("")

        return "\n".join(log_lines)
