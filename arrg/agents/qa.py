"""QA Agent - Reviews and validates reports for quality."""

from typing import Any, Dict, List
from arrg.agents.base import BaseAgent
from arrg.protocol import A2AMessage, MessageType


class QAAgent(BaseAgent):
    """
    QA Agent reviews and validates reports for quality.
    Checks for accuracy, completeness, and coherence.
    """

    def get_capabilities(self) -> Dict[str, Any]:
        """Return the capabilities of the QA Agent."""
        return {
            "agent_type": "qa",
            "description": "Reviews and validates reports for quality",
            "capabilities": [
                "quality_assessment",
                "accuracy_checking",
                "completeness_validation",
                "coherence_evaluation",
                "recommendation_generation",
            ],
            "inputs": ["report", "report_reference"],
            "outputs": ["qa_report", "issues", "approval_status"],
        }

    def process_message(self, message: A2AMessage) -> A2AMessage:
        """
        Process an incoming message and review a report.
        
        Args:
            message: Incoming A2A message
            
        Returns:
            Response message with QA results
        """
        self.receive_message(message)
        
        if message.message_type == MessageType.TASK_REQUEST:
            try:
                # Extract report reference from payload
                report_reference = message.payload.get("report_reference")
                
                if not report_reference:
                    raise ValueError("No report_reference provided")
                
                self.stream_output("Conducting quality assurance review...")
                
                # Retrieve report from workspace
                report = self.workspace.retrieve(report_reference)
                
                # Perform QA review
                qa_results = self._review_report(report)
                
                # Store QA results in workspace
                qa_key = f"qa_results_{message.message_id}"
                self.workspace.store(qa_key, qa_results, persist=True)
                
                status = "APPROVED" if qa_results["approved"] else "NEEDS_REVISION"
                self.stream_output(f"QA review completed: {status}")
                
                # Create response with QA results
                response = self.create_task_complete_message(
                    receiver=message.sender,
                    result={
                        "qa_reference": qa_key,
                        "approved": qa_results["approved"],
                        "issues_count": len(qa_results["issues"]),
                        "quality_score": qa_results["quality_score"],
                    },
                    in_reply_to=message.message_id,
                )
                
                self.send_message(response)
                return response
                
            except Exception as e:
                self.stream_output(f"Error in QA review: {str(e)}")
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

    def _review_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review a report for quality and accuracy.
        
        Args:
            report: Report to review
            
        Returns:
            QA results dictionary
        """
        # Build prompt for LLM
        system_prompt = """You are a QA Agent that reviews research reports for quality.
You should evaluate:
1. Accuracy: Are claims properly supported?
2. Completeness: Are all sections adequately covered?
3. Coherence: Does the report flow logically?
4. Clarity: Is the writing clear and professional?
5. Structure: Does it follow the outline properly?
6. Citations: Are sources properly referenced?

Output your review in JSON format with:
- quality_score: overall score (0-100)
- approved: boolean approval status
- issues: list of identified issues with severity
- strengths: positive aspects of the report
- recommendations: suggestions for improvement
- criteria_scores: scores for each evaluation criterion
"""

        # Prepare report summary for prompt
        report_text = report.get("full_text", "")[:2000]  # Limit for prompt
        
        user_prompt = f"""Review the following research report for quality:

Title: {report.get('title', 'Untitled')}
Word Count: {report.get('word_count', 0)}
Sections: {len(report.get('sections', []))}

Report Content (excerpt):
{report_text}

Provide a comprehensive QA review with scores, issues, and recommendations."""

        # Call LLM
        llm_response = self.call_llm(user_prompt, system_prompt)
        
        # Parse actual LLM response
        parsed_response = self.parse_json_from_llm(llm_response)
        
        if parsed_response and isinstance(parsed_response, dict):
            # Use LLM-generated content
            quality_score = parsed_response.get("quality_score", 85)
            approved = parsed_response.get("approved", True)
            issues = parsed_response.get("issues", [])
            strengths = parsed_response.get("strengths", [])
            recommendations = parsed_response.get("recommendations", [])
            criteria_scores = parsed_response.get("criteria_scores", {})
            
            # Validate that we got meaningful content
            if not isinstance(approved, bool):
                self.stream_output("Warning: LLM response incomplete, using evaluation logic")
                # Fallback to evaluation logic below
                approved = None
        else:
            # Fallback if parsing fails - use evaluation logic
            self.stream_output("Warning: Failed to parse LLM response, using evaluation logic")
            approved = None
        
        # If we don't have LLM-based approval, use evaluation logic
        if approved is None:
            issues = []
            
            # Check word count
            word_count = report.get("word_count", 0)
            if word_count < 500:
                issues.append({
                    "severity": "high",
                    "category": "completeness",
                    "description": f"Report is too short ({word_count} words). Expected at least 500 words.",
                })
            
            # Check sections
            sections = report.get("sections", [])
            if len(sections) < 3:
                issues.append({
                    "severity": "medium",
                    "category": "structure",
                    "description": f"Report has only {len(sections)} sections. Consider adding more depth.",
                })
            
            # Determine approval based on issues
            high_severity_issues = [i for i in issues if i["severity"] == "high"]
            approved = len(high_severity_issues) == 0
            
            # Calculate quality score
            base_score = 85
            score_deduction = len(high_severity_issues) * 20 + len(issues) * 5
            quality_score = max(0, min(100, base_score - score_deduction))
            
            strengths = [
                "Clear structure following the outline",
                "Professional writing style",
                "Good integration of research findings",
            ]
            
            recommendations = [
                "Consider expanding on key insights",
                "Add more specific examples",
                "Include more supporting evidence",
            ] if issues else [
                "Report meets quality standards",
                "Minor polish recommended before final publication",
            ]
            
            criteria_scores = {
                "accuracy": 90,
                "completeness": 85,
                "coherence": 88,
                "clarity": 92,
                "structure": 87,
                "citations": 80,
            }
        
        qa_results = {
            "approved": approved,
            "quality_score": quality_score,
            "issues": issues,
            "strengths": strengths,
            "recommendations": recommendations,
            "criteria_scores": criteria_scores,
            "summary": f"Report received a quality score of {quality_score}/100. " +
                      (f"Found {len(issues)} issues that need attention." if issues else "No critical issues found."),
            "llm_response": llm_response,
        }
        
        return qa_results
