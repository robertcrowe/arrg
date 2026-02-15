"""Writing Agent - Composes research reports from analysis.

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


class WritingAgent(BaseAgent):
    """
    Writing Agent composes research reports from analysis.
    Produces structured, well-written reports with proper sections.

    A2A Protocol: Receives a Task with analysis/plan references in the user
    Message's DataPart, produces an Artifact containing the report.
    Also handles revision Tasks where the Task contains QA feedback.
    """

    def get_capabilities(self) -> Dict[str, Any]:
        """Return the capabilities of the Writing Agent."""
        return {
            "agent_type": "writing",
            "description": "Composes research reports from analysis",
            "capabilities": [
                "report_composition",
                "section_writing",
                "narrative_structuring",
                "citation_integration",
                "report_revision",
            ],
            "inputs": ["analysis", "plan_reference", "qa_feedback"],
            "outputs": ["report", "sections", "full_text"],
        }

    def process_task(self, task: Task, message: Message) -> Task:
        """
        Process an A2A Task to write or revise a report.

        If the message data contains 'qa_feedback', this is a revision task.
        Otherwise, it's an initial writing task.

        Args:
            task: A2A Task (in SUBMITTED state)
            message: User Message containing analysis references or QA feedback

        Returns:
            Updated Task with report Artifact (COMPLETED or FAILED)
        """
        self.receive_message(message)
        task.add_to_history(message)

        try:
            data = message.get_data() or {}

            # Check if this is a revision task
            if "qa_feedback" in data:
                task.update_state(TaskState.WORKING, message="Revising report based on QA feedback")
                self.stream_output("Revising report based on QA feedback...")
                report = self._revise_report(data)
            else:
                task.update_state(TaskState.WORKING, message="Composing research report")
                self.stream_output("Composing research report...")
                report = self._write_report(data)

            # Store report in workspace
            report_key = f"report_{task.id}"
            self.workspace.store(report_key, report, persist=True)

            self.stream_output("Report completed successfully")

            # Complete the task with result
            result = {
                "report_reference": report_key,
                "word_count": len(report.get("full_text", "").split()),
                "section_count": len(report.get("sections", {})),
            }
            return self.create_completed_task(
                task, result_data=result,
                result_text="Report completed successfully",
            )

        except Exception as e:
            self.stream_output(f"Error writing report: {str(e)}")
            return self.create_failed_task(task, error=str(e))

    def _write_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write a research report from analysis data.

        Args:
            data: Message data containing references to plan and analysis

        Returns:
            Report dictionary with sections and full_text
        """
        plan_reference = data.get("plan_reference")
        analysis_reference = data.get("analysis_reference")

        # Retrieve plan and analysis from workspace
        plan = self.workspace.retrieve(plan_reference) if plan_reference else {}
        analysis = self.workspace.retrieve(analysis_reference) if analysis_reference else {}

        # Build prompt for LLM
        system_prompt = """You are a Writing Agent that composes comprehensive research reports.
You must write a well-structured report with:
1. Clear introduction with context and objectives
2. Well-organized sections following the outline
3. Integration of research findings and analysis insights
4. Proper conclusions and recommendations
5. Professional writing style

Output your report in JSON format with:
- title: report title
- sections: ordered dict of section_name -> section_content (full markdown text)
- full_text: the complete report as a single markdown document
- executive_summary: brief summary of findings
"""

        # Prepare context
        outline = plan.get("outline", {}) if plan else {}
        key_findings = analysis.get("key_findings", []) if analysis else []
        insights = analysis.get("insights", []) if analysis else []
        recommendations = analysis.get("recommendations", []) if analysis else []

        # Format outline
        if isinstance(outline, dict):
            outline_text = "\n".join(
                f"- {k}: {v}" if not isinstance(v, dict)
                else f"- {k}:\n" + "\n".join(f"  - {sk}: {sv}" for sk, sv in v.items())
                for k, v in outline.items()
            )
        elif isinstance(outline, list):
            outline_text = "\n".join(f"- {item}" for item in outline)
        else:
            outline_text = str(outline)

        # Format insights
        insights_text = ""
        if isinstance(insights, list):
            for i, insight in enumerate(insights):
                if isinstance(insight, dict):
                    insights_text += f"- {insight.get('title', f'Insight {i+1}')}: {insight.get('description', '')}\n"
                else:
                    insights_text += f"- {insight}\n"
        elif isinstance(insights, dict):
            for k, v in insights.items():
                insights_text += f"- {k}: {v}\n"

        user_prompt = f"""Write a comprehensive research report based on the following:

Topic: {plan.get('topic', 'Research Topic') if plan else 'Research Topic'}

Outline:
{outline_text}

Key Findings:
{chr(10).join(f'- {f}' for f in key_findings) if key_findings else '- No specific findings provided'}

Insights:
{insights_text or '- No specific insights provided'}

Recommendations:
{chr(10).join(f'- {r}' for r in recommendations) if recommendations else '- No specific recommendations provided'}

Write a professional, well-structured report following the outline."""

        # Call LLM with higher token limit for report generation
        llm_response = self.call_llm(user_prompt, system_prompt, max_tokens=16384)

        # Parse actual LLM response
        parsed_response = self.parse_json_from_llm(llm_response)

        if parsed_response and isinstance(parsed_response, dict):
            title = parsed_response.get("title", "Research Report")
            sections = parsed_response.get("sections", {})
            full_text = parsed_response.get("full_text", "")
            executive_summary = parsed_response.get("executive_summary", "")

            if not full_text and sections:
                full_text = f"# {title}\n\n"
                if executive_summary:
                    full_text += f"## Executive Summary\n\n{executive_summary}\n\n"
                for section_name, section_content in sections.items():
                    full_text += f"## {section_name}\n\n{section_content}\n\n"

            if not full_text:
                self.stream_output("Warning: LLM response incomplete, using raw response as report")
                full_text = llm_response
                title = "Research Report"
                sections = {"Full Report": llm_response}
        else:
            self.stream_output("Warning: Failed to parse LLM response, using raw response as report")
            title = "Research Report"
            full_text = llm_response
            sections = {"Full Report": llm_response}
            executive_summary = ""

        report = {
            "title": title,
            "sections": sections,
            "full_text": full_text,
            "executive_summary": executive_summary,
            "llm_response": llm_response,
        }

        return report

    def _revise_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Revise a report based on QA feedback.

        Args:
            data: Message data containing report reference and QA feedback

        Returns:
            Revised report dictionary
        """
        report_reference = data.get("report_reference")
        qa_feedback = data.get("qa_feedback", {})

        # Retrieve original report from workspace
        original_report = self.workspace.retrieve(report_reference) if report_reference else {}

        # Build revision prompt
        system_prompt = """You are a Writing Agent revising a research report based on QA feedback.
Address all issues raised by the QA Agent while maintaining the report's strengths.

Output the revised report in JSON format with:
- title: report title
- sections: ordered dict of section_name -> section_content
- full_text: the complete revised report as markdown
- executive_summary: brief summary
- revision_notes: what was changed
"""

        # Format QA issues
        issues = qa_feedback.get("issues", [])
        if isinstance(issues, list):
            issues_text = "\n".join(f"- {issue}" for issue in issues)
        elif isinstance(issues, dict):
            issues_text = "\n".join(f"- {k}: {v}" for k, v in issues.items())
        else:
            issues_text = str(issues)

        suggestions = qa_feedback.get("suggestions", [])
        if isinstance(suggestions, list):
            suggestions_text = "\n".join(f"- {s}" for s in suggestions)
        elif isinstance(suggestions, dict):
            suggestions_text = "\n".join(f"- {k}: {v}" for k, v in suggestions.items())
        else:
            suggestions_text = str(suggestions)

        user_prompt = f"""Revise the following report based on QA feedback:

Original Report:
{original_report.get('full_text', 'No report available')[:8000]}

QA Score: {qa_feedback.get('quality_score', 'N/A')}

Issues Found:
{issues_text}

Suggestions:
{suggestions_text}

Please address all issues and improve the report quality."""

        # Call LLM
        llm_response = self.call_llm(user_prompt, system_prompt, max_tokens=16384)

        # Parse actual LLM response
        parsed_response = self.parse_json_from_llm(llm_response)

        if parsed_response and isinstance(parsed_response, dict):
            title = parsed_response.get("title", original_report.get("title", "Research Report"))
            sections = parsed_response.get("sections", {})
            full_text = parsed_response.get("full_text", "")
            executive_summary = parsed_response.get("executive_summary", "")

            if not full_text and sections:
                full_text = f"# {title}\n\n"
                if executive_summary:
                    full_text += f"## Executive Summary\n\n{executive_summary}\n\n"
                for section_name, section_content in sections.items():
                    full_text += f"## {section_name}\n\n{section_content}\n\n"

            if not full_text:
                full_text = llm_response
                sections = {"Full Report": llm_response}
        else:
            title = original_report.get("title", "Research Report")
            full_text = llm_response
            sections = {"Full Report": llm_response}
            executive_summary = ""

        report = {
            "title": title,
            "sections": sections,
            "full_text": full_text,
            "executive_summary": executive_summary,
            "revision_notes": parsed_response.get("revision_notes", "Revised based on QA feedback") if parsed_response else "Revised based on QA feedback",
            "llm_response": llm_response,
        }

        return report
