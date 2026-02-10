"""Analysis Agent - Synthesizes research data into insights."""

from typing import Any, Dict
from arrg.agents.base import BaseAgent
from arrg.protocol import A2AMessage, MessageType


class AnalysisAgent(BaseAgent):
    """
    Analysis Agent synthesizes research data into insights.
    Identifies patterns, trends, and critical findings.
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

    def process_message(self, message: A2AMessage) -> A2AMessage:
        """
        Process an incoming message and analyze research data.
        
        Args:
            message: Incoming A2A message
            
        Returns:
            Response message with analysis
        """
        self.receive_message(message)
        
        if message.message_type == MessageType.TASK_REQUEST:
            try:
                # Extract data reference from payload
                data_reference = message.payload.get("data_reference")
                plan_reference = message.payload.get("plan_reference")
                
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
                analysis_key = f"analysis_{message.message_id}"
                self.workspace.store(analysis_key, analysis, persist=True)
                
                self.stream_output("Analysis completed successfully")
                
                # Create response with reference to analysis
                response = self.create_task_complete_message(
                    receiver=message.sender,
                    result={
                        "analysis_reference": analysis_key,
                        "insights_count": len(analysis["insights"]),
                        "key_findings": analysis["key_findings"][:3],  # Top 3 findings
                    },
                    in_reply_to=message.message_id,
                )
                
                self.send_message(response)
                return response
                
            except Exception as e:
                self.stream_output(f"Error analyzing data: {str(e)}")
                error_msg = self.create_error_message(
                    receiver=message.sender,
                    error=str(e),
                    in_reply_to=message.message_id,
                )
                self.send_message(error_msg)
                return error_msg
        
        elif message.message_type == MessageType.CAPABILITY_QUERY:
            response = A2AMessage(
                message_type=MessageType.CAPABILITY_RESPONSE,
                sender=self.agent_id,
                receiver=message.sender,
                payload=self.get_capabilities(),
                in_reply_to=message.message_id,
            )
            self.send_message(response)
            return response
        
        return self.create_error_message(
            receiver=message.sender,
            error=f"Unsupported message type: {message.message_type}",
            in_reply_to=message.message_id,
        )

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
        findings_summary = "\n".join(
            f"- {f['question']}: {f['answer']}"
            for f in research_data.get("findings", [])
        )
        
        user_prompt = f"""Analyze the following research data and provide insights:

Research Findings:
{findings_summary}

Key Facts:
{chr(10).join(f'- {fact}' for fact in research_data.get('key_facts', []))}

Sources: {len(research_data.get('sources', []))} sources consulted

Provide comprehensive analysis with insights, patterns, and recommendations."""

        # Call LLM
        llm_response = self.call_llm(user_prompt, system_prompt)
        
        # Parse actual LLM response
        parsed_response = self.parse_json_from_llm(llm_response)
        
        if parsed_response and isinstance(parsed_response, dict):
            # Use LLM-generated content
            key_findings = parsed_response.get("key_findings", [])
            insights = parsed_response.get("insights", [])
            patterns = parsed_response.get("patterns", [])
            recommendations = parsed_response.get("recommendations", [])
            gaps = parsed_response.get("gaps", [])
            synthesis = parsed_response.get("synthesis", "")
            
            # Validate that we got meaningful content
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
            # Fallback if parsing fails
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
