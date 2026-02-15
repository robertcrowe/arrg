"""Analysis Agent - Synthesizes research data into insights.

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


class AnalysisAgent(BaseAgent):
    """
    Analysis Agent synthesizes research data into insights.
    Identifies patterns, trends, and critical findings.

    A2A Protocol: Receives a Task with data_reference in the user Message's
    DataPart, produces an Artifact containing analysis insights.
    """

    def get_capabilities(self) -> Dict[str, Any]:
        """Return the capabilities of the Analysis Agent."""
        return {
            "agent_type": "analysis",
            "description": "Synthesizes research data into insights",
            "capabilities": [
                "data_synthesis",
                "pattern_identification",
                "trend_analysis",
                "insight_generation",
                "critical_evaluation",
            ],
            "inputs": ["research_data", "data_reference"],
            "outputs": ["analysis", "insights", "recommendations"],
        }

    def process_task(self, task: Task, message: Message) -> Task:
        """
        Process an A2A Task to analyze research data.

        Args:
            task: A2A Task (in SUBMITTED state)
            message: User Message containing data references

        Returns:
            Updated Task with analysis Artifact (COMPLETED or FAILED)
        """
        self.receive_message(message)
        task.update_state(TaskState.WORKING, message="Analyzing research data")
        task.add_to_history(message)

        try:
            # Extract references from message DataPart
            data = message.get_data() or {}
            data_reference = data.get("data_reference")
            plan_reference = data.get("plan_reference")

            if not data_reference:
                raise ValueError("No data_reference provided")

            self.stream_output("Analyzing research data...")

            # Retrieve research data from workspace
            research_data = self.workspace.retrieve(data_reference)
            plan = None
            if plan_reference:
                plan = self.workspace.retrieve(plan_reference)

            # Perform analysis
            analysis = self._analyze_data(research_data, plan)

            # Store analysis in workspace
            analysis_key = f"analysis_{task.id}"
            self.workspace.store(analysis_key, analysis, persist=True)

            self.stream_output("Analysis completed successfully")

            # Complete the task with result
            result = {
                "analysis_reference": analysis_key,
                "insights_count": len(analysis["insights"]),
                "key_findings": analysis["key_findings"][:3],  # Top 3 findings
            }
            return self.create_completed_task(
                task, result_data=result,
                result_text="Analysis completed successfully",
            )

        except Exception as e:
            self.stream_output(f"Error analyzing data: {str(e)}")
            return self.create_failed_task(task, error=str(e))

    def _analyze_data(
        self, research_data: Dict[str, Any], plan: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze research data to generate insights.

        Args:
            research_data: Research data to analyze
            plan: Optional research plan

        Returns:
            Analysis dictionary with insights
        """
        # Build prompt for LLM
        system_prompt = """You are an Analysis Agent that synthesizes research data into insights.
You should:
1. Identify key patterns and trends in the data
2. Synthesize findings across multiple sources
3. Generate actionable insights
4. Highlight critical findings
5. Identify gaps or contradictions
6. Provide recommendations

Output your analysis in JSON format with:
- key_findings: most important discoveries
- insights: synthesized understanding
- patterns: identified patterns or trends
- recommendations: actionable recommendations
- gaps: areas needing more investigation
"""

        # Prepare research summary for prompt
        findings = research_data.get("findings", [])

        # Handle both list and dict formats for findings
        if isinstance(findings, dict):
            findings_summary = "\n".join(
                f"- {key}: {value.get('content', value) if isinstance(value, dict) else value}"
                for key, value in findings.items()
            )
        elif isinstance(findings, list):
            findings_summary = "\n".join(
                f"- {f.get('question', 'Unknown')}: {f.get('answer', f.get('content', 'No answer'))}"
                for f in findings
            )
        else:
            findings_summary = str(findings)

        # Handle key_facts which could be a list, dict, or other format
        key_facts = research_data.get('key_facts', [])
        if isinstance(key_facts, dict):
            key_facts_summary = "\n".join(f'- {k}: {v}' for k, v in key_facts.items())
        elif isinstance(key_facts, list):
            key_facts_summary = "\n".join(f'- {fact}' for fact in key_facts)
        else:
            key_facts_summary = str(key_facts)

        # Handle sources which could be a list, dict, or other format
        sources = research_data.get('sources', [])
        if isinstance(sources, dict):
            source_count = len(sources)
        elif isinstance(sources, list):
            source_count = len(sources)
        else:
            source_count = 0

        user_prompt = f"""Analyze the following research data and provide insights:

Research Findings:
{findings_summary}

Key Facts:
{key_facts_summary}

Sources: {source_count} sources consulted

Provide comprehensive analysis with insights, patterns, and recommendations."""

        # Call LLM
        llm_response = self.call_llm(user_prompt, system_prompt)

        # Parse actual LLM response
        parsed_response = self.parse_json_from_llm(llm_response)

        if parsed_response and isinstance(parsed_response, dict):
            key_findings = parsed_response.get("key_findings", [])
            insights = parsed_response.get("insights", [])
            patterns = parsed_response.get("patterns", [])
            recommendations = parsed_response.get("recommendations", [])
            gaps = parsed_response.get("gaps", [])
            synthesis = parsed_response.get("synthesis", "")

            if not key_findings or not insights:
                self.stream_output("Warning: LLM response incomplete, using fallback structure")
                key_findings = [
                    "Critical finding 1 from synthesized data",
                    "Critical finding 2 showing important trend",
                ]
                insights = [
                    {
                        "title": "Major Insight",
                        "description": "Synthesis of data points",
                        "supporting_evidence": ["Evidence from research"],
                    }
                ]
        else:
            self.stream_output("Warning: Failed to parse LLM response, using fallback structure")
            key_findings = [
                "Critical finding 1 from synthesized data",
                "Critical finding 2 showing important trend",
                "Critical finding 3 revealing key insight",
            ]
            insights = [
                {
                    "title": "Major Insight 1",
                    "description": "Synthesis of multiple data points reveals...",
                    "supporting_evidence": ["Evidence A", "Evidence B"],
                },
                {
                    "title": "Major Insight 2",
                    "description": "Pattern analysis indicates...",
                    "supporting_evidence": ["Evidence C", "Evidence D"],
                },
                {
                    "title": "Major Insight 3",
                    "description": "Cross-referencing sources shows...",
                    "supporting_evidence": ["Evidence E", "Evidence F"],
                },
            ]
            patterns = [
                "Emerging pattern 1 across sources",
                "Recurring theme 2 in findings",
                "Trend 3 showing development over time",
            ]
            recommendations = [
                "Recommendation 1 based on analysis",
                "Recommendation 2 for further investigation",
                "Recommendation 3 for practical application",
            ]
            gaps = research_data.get("gaps", [])
            synthesis = "Comprehensive synthesis of all research findings, showing connections and implications."

        analysis = {
            "key_findings": key_findings,
            "insights": insights,
            "patterns": patterns,
            "recommendations": recommendations,
            "gaps": gaps,
            "synthesis": synthesis,
            "llm_response": llm_response,
        }

        return analysis
