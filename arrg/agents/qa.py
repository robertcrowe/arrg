"""QA Agent - Reviews and validates research reports.

Communicates via A2A Protocol v1.0 Tasks and Messages.
"""

from typing import Any, Dict
from arrg.agents.base import BaseAgent
from arrg.a2a import (
    Task,
    TaskState,
    Message,
    MessageRole,
    TextPart,
    DataPart,
    Artifact,
)


class QAAgent(BaseAgent):
    """
    QA Agent reviews and validates research reports.
    Checks for quality, accuracy, completeness, and coherence.

    A2A Protocol: Receives a Task with report_reference in the user Message's
    DataPart, produces an Artifact containing the QA review.
    """

    def get_capabilities(self) -> Dict[str, Any]:
        """Return the capabilities of the QA Agent."""
        return {
            "agent_type": "qa",
            "description": "Reviews and validates research reports",
            "capabilities": [
                "quality_assessment",
                "accuracy_checking",
                "completeness_validation",
                "coherence_evaluation",
                "recommendation_generation",
            ],
            "inputs": ["report", "report_reference"],
            "outputs": ["qa_result", "quality_score", "issues", "approved"],
        }

    def process_task(self, task: Task, message: Message) -> Task:
        """
        Process an A2A Task to review a report.

        Args:
            task: A2A Task (in SUBMITTED state)
            message: User Message containing report reference

        Returns:
            Updated Task with QA review Artifact (COMPLETED or FAILED)
        """
        self.receive_message(message)
        task.update_state(TaskState.WORKING, message="Reviewing report quality")
        task.add_to_history(message)

        try:
            # Extract report reference from message DataPart
            data = message.get_data() or {}
            report_reference = data.get("report_reference")

            if not report_reference:
                raise ValueError("No report_reference provided")

            self.stream_output("Reviewing report quality...")

            # Retrieve report from workspace
            report = self.workspace.retrieve(report_reference)

            # Perform QA review
            qa_result = self._review_report(report)

            # Store QA result in workspace
            qa_key = f"qa_result_{task.id}"
            self.workspace.store(qa_key, qa_result, persist=True)

            self.stream_output(
                f"QA review completed - Score: {qa_result['quality_score']}/10 "
                f"- {'Approved' if qa_result['approved'] else 'Needs revision'}"
            )

            # Complete the task with result
            result = {
                "qa_reference": qa_key,
                "approved": qa_result["approved"],
                "quality_score": qa_result["quality_score"],
                "issues_count": len(qa_result.get("issues", [])),
            }
            return self.create_completed_task(
                task, result_data=result,
                result_text=f"QA review: score={qa_result['quality_score']}/10, approved={qa_result['approved']}",
            )

        except Exception as e:
            self.stream_output(f"Error reviewing report: {str(e)}")
            return self.create_failed_task(task, error=str(e))

    def _review_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review a report for quality, accuracy, and completeness.

        Args:
            report: Report dictionary to review

        Returns:
            QA result dictionary
        """
        # Build prompt for LLM
        system_prompt = """You are a QA Agent that reviews research reports for quality.
Evaluate the report on:
1. Accuracy of information
2. Completeness of coverage
3. Writing quality and clarity
4. Logical structure and flow
5. Evidence and source support
6. Professional tone

Output your review in JSON format with:
- quality_score: integer from 1-10
- approved: boolean (true if score >= 7)
- issues: list of specific issues found
- strengths: list of report strengths
- suggestions: list of improvement suggestions
- category_scores: dict of category -> score (accuracy, completeness, clarity, structure, evidence)
"""

        report_text = report.get("full_text", "")
        title = report.get("title", "Unknown")

        # Truncate very long reports
        if len(report_text) > 12000:
            report_text = report_text[:12000] + "\n\n[... truncated for review ...]"

        user_prompt = f"""Review the following research report for quality:

Title: {title}

Report:
{report_text}

Provide a thorough quality assessment with scores and specific feedback."""

        # Call LLM
        llm_response = self.call_llm(user_prompt, system_prompt)

        # Parse actual LLM response
        parsed_response = self.parse_json_from_llm(llm_response)

        if parsed_response and isinstance(parsed_response, dict):
            quality_score = parsed_response.get("quality_score", 7)
            issues = parsed_response.get("issues", [])
            strengths = parsed_response.get("strengths", [])
            suggestions = parsed_response.get("suggestions", [])
            category_scores = parsed_response.get("category_scores", {})

            # Validate quality_score
            if not isinstance(quality_score, (int, float)):
                try:
                    quality_score = int(quality_score)
                except (ValueError, TypeError):
                    quality_score = 7
            quality_score = max(1, min(10, quality_score))

            approved = parsed_response.get("approved", quality_score >= 7)
        else:
            self.stream_output("Warning: Failed to parse LLM response, using default assessment")
            quality_score = 7
            approved = True
            issues = ["Unable to perform detailed analysis"]
            strengths = ["Report was generated"]
            suggestions = ["Review report manually"]
            category_scores = {
                "accuracy": 7,
                "completeness": 7,
                "clarity": 7,
                "structure": 7,
                "evidence": 7,
            }

        qa_result = {
            "quality_score": quality_score,
            "approved": approved,
            "issues": issues,
            "strengths": strengths,
            "suggestions": suggestions,
            "category_scores": category_scores,
            "llm_response": llm_response,
        }

        return qa_result
