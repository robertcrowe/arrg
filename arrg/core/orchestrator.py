"""Orchestrator for managing the multi-agent workflow."""

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
from arrg.protocol import A2AMessage, MessageType, SharedWorkspace, TaskStatus
from arrg.a2a import Task, TaskStatus as A2ATaskStatus


class Orchestrator:
    """
    Orchestrates the multi-agent research report generation workflow.
    Manages agent communication and workflow coordination.
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
            models: Dict mapping agent names to model strings. If None, uses default model for all.
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
        
        # Initialize shared workspace
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
        
        self.current_status = TaskStatus.PENDING
        self.workflow_progress = {
            "planning": TaskStatus.PENDING,
            "research": TaskStatus.PENDING,
            "analysis": TaskStatus.PENDING,
            "writing": TaskStatus.PENDING,
            "qa": TaskStatus.PENDING,
        }
        
        # A2A message history for logging
        self.message_history: list[A2AMessage] = []
        
        # A2A Task tracking
        self.tasks: Dict[str, Task] = {}
        
        # QA revision tracking
        self.max_qa_retries = 2
        self.qa_retry_count = 0

    def generate_report(self, topic: str, requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a research report on the given topic.
        
        Args:
            topic: Research topic
            requirements: Optional requirements and constraints
            
        Returns:
            Final report and metadata
        """
        self.logger.info(f"Starting report generation for topic: {topic}")
        self.stream_output(f"Starting report generation for: {topic}")
        self.current_status = TaskStatus.IN_PROGRESS
        
        try:
            # Step 1: Planning
            self.stream_output("\n=== PHASE 1: PLANNING ===")
            self.workflow_progress["planning"] = TaskStatus.IN_PROGRESS
            plan_result = self._execute_planning(topic, requirements or {})
            self.workflow_progress["planning"] = TaskStatus.COMPLETE
            
            # Step 2: Research
            self.stream_output("\n=== PHASE 2: RESEARCH ===")
            self.workflow_progress["research"] = TaskStatus.IN_PROGRESS
            research_result = self._execute_research(plan_result)
            self.workflow_progress["research"] = TaskStatus.COMPLETE
            
            # Step 3: Analysis
            self.stream_output("\n=== PHASE 3: ANALYSIS ===")
            self.workflow_progress["analysis"] = TaskStatus.IN_PROGRESS
            analysis_result = self._execute_analysis(research_result, plan_result)
            self.workflow_progress["analysis"] = TaskStatus.COMPLETE
            
            # Step 4: Writing with QA revision loop
            self.stream_output("\n=== PHASE 4: WRITING ===")
            self.workflow_progress["writing"] = TaskStatus.IN_PROGRESS
            writing_result = self._execute_writing(analysis_result, plan_result)
            self.workflow_progress["writing"] = TaskStatus.COMPLETE
            
            # Step 5: QA with revision loop
            self.stream_output("\n=== PHASE 5: QUALITY ASSURANCE ===")
            self.workflow_progress["qa"] = TaskStatus.IN_PROGRESS
            self.qa_retry_count = 0
            qa_result = None
            
            while self.qa_retry_count <= self.max_qa_retries:
                qa_result = self._execute_qa(writing_result)
                
                # Check if QA approved the report
                if qa_result.get("approved", False):
                    self.stream_output("✓ QA Review: Report APPROVED")
                    self.workflow_progress["qa"] = TaskStatus.COMPLETE
                    break
                else:
                    self.qa_retry_count += 1
                    if self.qa_retry_count <= self.max_qa_retries:
                        self.stream_output(
                            f"✗ QA Review: Report REJECTED (Attempt {self.qa_retry_count}/{self.max_qa_retries})"
                        )
                        self.stream_output("→ Sending back to Writing Agent for revision...")
                        
                        # Send rejection back to Writing Agent with QA feedback
                        self.workflow_progress["writing"] = TaskStatus.IN_PROGRESS
                        writing_result = self._execute_writing_revision(
                            analysis_result, plan_result, qa_result
                        )
                        self.workflow_progress["writing"] = TaskStatus.COMPLETE
                    else:
                        self.stream_output(
                            f"✗ QA Review: Report REJECTED (Max retries reached)"
                        )
                        self.stream_output("→ Proceeding with current version despite issues...")
                        self.workflow_progress["qa"] = TaskStatus.COMPLETE
                        break
            
            self.current_status = TaskStatus.COMPLETE
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
            self.current_status = TaskStatus.ERROR
            return {
                "status": "error",
                "error": str(e),
                "workflow_status": self.workflow_progress,
            }

    def _execute_planning(self, topic: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planning phase using A2A Task."""
        # Create A2A Task
        task = Task(
            task_id=f"planning_{len(self.tasks)}",
            description=f"Create research plan for topic: {topic}",
            assigned_agent="planning",
            metadata={"topic": topic, "requirements": requirements}
        )
        task.update_status(A2ATaskStatus.IN_PROGRESS)
        self.tasks[task.task_id] = task
        
        # Create A2A message for task request
        message = A2AMessage(
            message_type=MessageType.TASK_REQUEST,
            sender="orchestrator",
            receiver="planning",
            payload={
                "task_id": task.task_id,
                "topic": topic,
                "requirements": requirements,
            },
        )
        
        self.message_history.append(message)
        self.logger.info(f"Created task {task.task_id} for planning agent")
        
        try:
            response = self.agents["planning"].process_message(message)
            self.message_history.append(response)
            
            if response.message_type == MessageType.ERROR:
                task.update_status(A2ATaskStatus.FAILED, error=response.payload.get("error"))
                raise RuntimeError(f"Planning failed: {response.payload.get('error')}")
            
            # Mark task as complete
            task.set_result(response.payload)
            task.update_status(A2ATaskStatus.COMPLETED)
            return response.payload
            
        except Exception as e:
            task.update_status(A2ATaskStatus.FAILED, error=str(e))
            raise

    def _execute_research(self, plan_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research phase using A2A Task."""
        # Create A2A Task
        task = Task(
            task_id=f"research_{len(self.tasks)}",
            description="Conduct research on questions from plan",
            assigned_agent="research",
            metadata={
                "research_questions": plan_result["research_questions"],
                "plan_reference": plan_result["plan_reference"]
            }
        )
        task.update_status(A2ATaskStatus.IN_PROGRESS)
        self.tasks[task.task_id] = task
        
        message = A2AMessage(
            message_type=MessageType.TASK_REQUEST,
            sender="orchestrator",
            receiver="research",
            payload={
                "task_id": task.task_id,
                "research_questions": plan_result["research_questions"],
                "plan_reference": plan_result["plan_reference"],
            },
        )
        
        self.message_history.append(message)
        self.logger.info(f"Created task {task.task_id} for research agent")
        
        try:
            response = self.agents["research"].process_message(message)
            self.message_history.append(response)
            
            if response.message_type == MessageType.ERROR:
                task.update_status(A2ATaskStatus.FAILED, error=response.payload.get("error"))
                raise RuntimeError(f"Research failed: {response.payload.get('error')}")
            
            task.set_result(response.payload)
            task.update_status(A2ATaskStatus.COMPLETED)
            return response.payload
            
        except Exception as e:
            task.update_status(A2ATaskStatus.FAILED, error=str(e))
            raise

    def _execute_analysis(
        self, research_result: Dict[str, Any], plan_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the analysis phase using A2A Task."""
        # Create A2A Task
        task = Task(
            task_id=f"analysis_{len(self.tasks)}",
            description="Analyze research data and synthesize insights",
            assigned_agent="analysis",
            metadata={
                "data_reference": research_result["data_reference"],
                "plan_reference": plan_result["plan_reference"]
            }
        )
        task.update_status(A2ATaskStatus.IN_PROGRESS)
        self.tasks[task.task_id] = task
        
        message = A2AMessage(
            message_type=MessageType.TASK_REQUEST,
            sender="orchestrator",
            receiver="analysis",
            payload={
                "task_id": task.task_id,
                "data_reference": research_result["data_reference"],
                "plan_reference": plan_result["plan_reference"],
            },
        )
        
        self.message_history.append(message)
        self.logger.info(f"Created task {task.task_id} for analysis agent")
        
        try:
            response = self.agents["analysis"].process_message(message)
            self.message_history.append(response)
            
            if response.message_type == MessageType.ERROR:
                task.update_status(A2ATaskStatus.FAILED, error=response.payload.get("error"))
                raise RuntimeError(f"Analysis failed: {response.payload.get('error')}")
            
            task.set_result(response.payload)
            task.update_status(A2ATaskStatus.COMPLETED)
            return response.payload
            
        except Exception as e:
            task.update_status(A2ATaskStatus.FAILED, error=str(e))
            raise

    def _execute_writing(
        self, analysis_result: Dict[str, Any], plan_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the writing phase using A2A Task."""
        # Create A2A Task
        task = Task(
            task_id=f"writing_{len(self.tasks)}",
            description="Write comprehensive research report",
            assigned_agent="writing",
            metadata={
                "analysis_reference": analysis_result["analysis_reference"],
                "plan_reference": plan_result["plan_reference"]
            }
        )
        task.update_status(A2ATaskStatus.IN_PROGRESS)
        self.tasks[task.task_id] = task
        
        message = A2AMessage(
            message_type=MessageType.TASK_REQUEST,
            sender="orchestrator",
            receiver="writing",
            payload={
                "task_id": task.task_id,
                "analysis_reference": analysis_result["analysis_reference"],
                "plan_reference": plan_result["plan_reference"],
            },
        )
        
        self.message_history.append(message)
        self.logger.info(f"Created task {task.task_id} for writing agent")
        
        try:
            response = self.agents["writing"].process_message(message)
            self.message_history.append(response)
            
            if response.message_type == MessageType.ERROR:
                task.update_status(A2ATaskStatus.FAILED, error=response.payload.get("error"))
                raise RuntimeError(f"Writing failed: {response.payload.get('error')}")
            
            task.set_result(response.payload)
            task.update_status(A2ATaskStatus.COMPLETED)
            return response.payload
            
        except Exception as e:
            task.update_status(A2ATaskStatus.FAILED, error=str(e))
            raise

    def _execute_qa(self, writing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the QA phase using A2A Task."""
        # Create A2A Task
        task = Task(
            task_id=f"qa_{len(self.tasks)}",
            description="Quality assurance review of report",
            assigned_agent="qa",
            metadata={"report_reference": writing_result["report_reference"]}
        )
        task.update_status(A2ATaskStatus.IN_PROGRESS)
        self.tasks[task.task_id] = task
        
        message = A2AMessage(
            message_type=MessageType.TASK_REQUEST,
            sender="orchestrator",
            receiver="qa",
            payload={
                "task_id": task.task_id,
                "report_reference": writing_result["report_reference"],
            },
        )
        
        self.message_history.append(message)
        self.logger.info(f"Created task {task.task_id} for qa agent")
        
        try:
            response = self.agents["qa"].process_message(message)
            self.message_history.append(response)
            
            if response.message_type == MessageType.ERROR:
                task.update_status(A2ATaskStatus.FAILED, error=response.payload.get("error"))
                raise RuntimeError(f"QA failed: {response.payload.get('error')}")
            
            task.set_result(response.payload)
            task.update_status(A2ATaskStatus.COMPLETED)
            return response.payload
            
        except Exception as e:
            task.update_status(A2ATaskStatus.FAILED, error=str(e))
            raise

    def _execute_writing_revision(
        self, 
        analysis_result: Dict[str, Any], 
        plan_result: Dict[str, Any],
        qa_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the writing phase with QA feedback for revision."""
        message = A2AMessage(
            message_type=MessageType.TASK_REJECTED,
            sender="orchestrator",
            receiver="writing",
            payload={
                "analysis_reference": analysis_result["analysis_reference"],
                "plan_reference": plan_result["plan_reference"],
                "qa_feedback": qa_result,
                "revision_required": True,
            },
        )
        
        self.message_history.append(message)
        response = self.agents["writing"].process_message(message)
        self.message_history.append(response)
        
        if response.message_type == MessageType.ERROR:
            raise RuntimeError(f"Writing revision failed: {response.payload.get('error')}")
        
        return response.payload

    def stream_output(self, text: str):
        """Stream output to the dashboard console."""
        if self.stream_callback:
            self.stream_callback(text)
        self.logger.info(text)

    def get_progress(self) -> Dict[str, Any]:
        """Get current workflow progress."""
        return {
            "status": self.current_status.value,
            "phases": self.workflow_progress,
        }

    def get_message_log(self) -> str:
        """
        Get a downloadable log of all A2A messages and agent outputs.
        
        Returns:
            Formatted log string
        """
        log_lines = [
            "=" * 80,
            "ARRG WORKFLOW LOG",
            "=" * 80,
            "",
            f"Models Configuration:",
        ]
        
        for agent, model in self.models.items():
            log_lines.append(f"  {agent}: {model}")
        
        log_lines.extend([
            "",
            "=" * 80,
            "A2A MESSAGE HISTORY",
            "=" * 80,
            "",
        ])
        
        for i, msg in enumerate(self.message_history, 1):
            log_lines.extend([
                f"Message {i}:",
                f"  ID: {msg.message_id}",
                f"  Type: {msg.message_type.value}",
                f"  From: {msg.sender}",
                f"  To: {msg.receiver}",
                f"  Timestamp: {msg.timestamp}",
                f"  In Reply To: {msg.in_reply_to or 'N/A'}",
                f"  Payload: {str(msg.payload)[:200]}...",
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
                "-" * 40,
            ])
            
            for i, msg in enumerate(agent.message_history, 1):
                log_lines.extend([
                    f"  [{i}] {msg.message_type.value}: {msg.sender} → {msg.receiver}",
                    f"      ID: {msg.message_id}",
                    f"      Timestamp: {msg.timestamp}",
                ])
            
            log_lines.append("")
        
        return "\n".join(log_lines)
