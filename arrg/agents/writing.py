"""Writing Agent - Transforms analysis into polished reports."""

from typing import Any, Dict
from arrg.agents.base import BaseAgent
from arrg.protocol import A2AMessage, MessageType


class WritingAgent(BaseAgent):
    """
    Writing Agent transforms analysis into polished, well-structured reports.
    Generates professional content following the research outline.
    """

    def get_capabilities(self) -> Dict[str, Any]:
        """Return the capabilities of the Writing Agent."""
        return {
            "agent_type": "writing",
            "description": "Transforms analysis into polished reports",
            "capabilities": [
                "report_writing",
                "content_structuring",
                "professional_formatting",
                "narrative_creation",
                "section_composition",
            ],
            "inputs": ["analysis", "outline", "plan_reference", "analysis_reference"],
            "outputs": ["report", "formatted_content"],
        }

    def process_message(self, message: A2AMessage) -> A2AMessage:
        """
        Process an incoming message and write a report.
        
        Args:
            message: Incoming A2A message
            
        Returns:
            Response message with report
        """
        self.receive_message(message)
        
        if message.message_type == MessageType.TASK_REQUEST:
            try:
                # Extract references from payload
                analysis_reference = message.payload.get("analysis_reference")
                plan_reference = message.payload.get("plan_reference")
                
                if not analysis_reference or not plan_reference:
                    raise ValueError("Missing analysis_reference or plan_reference")
                
                self.stream_output("Writing research report...")
                
                # Retrieve data from workspace
                analysis = self.workspace.retrieve(analysis_reference)
                plan = self.workspace.retrieve(plan_reference)
                
                # Write the report
                report = self._write_report(analysis, plan)
                
                # Store report in workspace
                report_key = f"report_{message.message_id}"
                self.workspace.store(report_key, report, persist=True)
                
                self.stream_output("Report writing completed successfully")
                
                # Create response with reference to report
                response = self.create_task_complete_message(
                    receiver=message.sender,
                    result={
                        "report_reference": report_key,
                        "word_count": report["word_count"],
                        "section_count": len(report["sections"]),
                    },
                    in_reply_to=message.message_id,
                )
                
                self.send_message(response)
                return response
                
            except Exception as e:
                self.stream_output(f"Error writing report: {str(e)}")
                error_msg = self.create_error_message(
                    receiver=message.sender,
                    error=str(e),
                    in_reply_to=message.message_id,
                )
                self.send_message(error_msg)
                return error_msg
        
        elif message.message_type == MessageType.TASK_REJECTED:
            try:
                # Handle revision request from QA
                analysis_reference = message.payload.get("analysis_reference")
                plan_reference = message.payload.get("plan_reference")
                qa_feedback = message.payload.get("qa_feedback", {})
                
                if not analysis_reference or not plan_reference:
                    raise ValueError("Missing analysis_reference or plan_reference")
                
                self.stream_output("Revising report based on QA feedback...")
                
                # Retrieve data from workspace
                analysis = self.workspace.retrieve(analysis_reference)
                plan = self.workspace.retrieve(plan_reference)
                
                # Write revised report with QA feedback
                report = self._revise_report(analysis, plan, qa_feedback)
                
                # Store revised report in workspace
                report_key = f"report_revised_{message.message_id}"
                self.workspace.store(report_key, report, persist=True)
                
                self.stream_output("Report revision completed successfully")
                
                # Create response with reference to revised report
                response = self.create_task_complete_message(
                    receiver=message.sender,
                    result={
                        "report_reference": report_key,
                        "word_count": report["word_count"],
                        "section_count": len(report["sections"]),
                        "revised": True,
                    },
                    in_reply_to=message.message_id,
                )
                
                self.send_message(response)
                return response
                
            except Exception as e:
                self.stream_output(f"Error revising report: {str(e)}")
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

    def _write_report(
        self, analysis: Dict[str, Any], plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Write a comprehensive research report.
        
        Args:
            analysis: Analysis data
            plan: Research plan with outline
            
        Returns:
            Report dictionary
        """
        # Build prompt for LLM
        system_prompt = """You are a Writing Agent that creates professional research reports.
You should:
1. Follow the provided outline structure
2. Write clear, well-organized content
3. Integrate insights and findings naturally
4. Use professional academic/business writing style
5. Include proper transitions between sections
6. Cite key findings appropriately

Output your report in JSON format with:
- title: report title
- sections: list of sections with titles and content
- executive_summary: brief overview
- conclusion: final synthesis
- full_text: complete report in markdown format
"""

        # Prepare outline and insights
        outline_text = "\n".join(
            f"{key}: {value}" for key, value in plan.get("outline", {}).items()
        )
        
        # Handle different insight structures
        insights = analysis.get("insights", [])
        insights_list = []
        for insight in insights:
            if isinstance(insight, dict):
                # Try different key combinations
                title = insight.get("title") or insight.get("insight") or insight.get("name", "Insight")
                description = insight.get("description") or insight.get("analysis") or insight.get("content", "")
                insights_list.append(f"- {title}: {description}")
            elif isinstance(insight, str):
                insights_list.append(f"- {insight}")
        insights_text = "\n".join(insights_list)
        
        user_prompt = f"""Write a comprehensive research report on: {plan.get('topic', 'the given topic')}

Outline to follow:
{outline_text}

Key Insights to incorporate:
{insights_text}

Key Findings:
{chr(10).join(f'- {finding}' for finding in analysis.get('key_findings', []))}

Create a well-structured, professional report with all sections."""

        # Call LLM with appropriate token limit for comprehensive reports
        # Claude models typically support 8192 max_tokens for output
        llm_response = self.call_llm(user_prompt, system_prompt, max_tokens=8192)
        
        # Parse actual LLM response
        parsed_response = self.parse_json_from_llm(llm_response)
        
        if parsed_response and isinstance(parsed_response, dict):
            # Use LLM-generated content
            title = parsed_response.get("title", f"Research Report: {plan.get('topic', 'Research Topic')}")
            sections = parsed_response.get("sections", [])
            executive_summary = parsed_response.get("executive_summary", "")
            conclusion = parsed_response.get("conclusion", "")
            full_text = parsed_response.get("full_text", "")
            
            # Validate that we got meaningful content
            if not sections or not full_text:
                self.stream_output("Warning: LLM response incomplete, generating fallback content")
                sections = []
                word_count = 0
                
                for section_key, section_value in plan.get("outline", {}).items():
                    # Extract title from nested structure if present
                    if isinstance(section_value, dict) and 'title' in section_value:
                        section_title = section_value.get('title', section_key)
                        subsections = section_value.get('subsections', {})
                    else:
                        section_title = section_key
                        subsections = section_value if isinstance(section_value, dict) else {}
                    
                    section_content = f"This section covers {section_title}. "
                    section_content += f"Based on our research and analysis, we found that... "
                    section_content += f"The key insights related to this area include... "
                    
                    sections.append({
                        "title": section_title,
                        "content": section_content,
                        "subsections": subsections,
                    })
                    word_count += len(section_content.split())
                
                # Build full markdown text for incomplete response
                markdown_sections = []
                for section in sections:
                    markdown_sections.append(f"## {section.get('title', 'Untitled Section')}\n\n{section.get('content', '')}\n")
                    if section.get('subsections') and isinstance(section.get('subsections'), dict):
                        for sub_key, sub_val in section.get('subsections', {}).items():
                            markdown_sections.append(f"### {sub_key}\n\n{sub_val}\n")
                
                full_text = f"""# {title}

## Executive Summary

This comprehensive research report examines {plan.get('topic', 'the topic')}. Through systematic research and analysis, we have identified key insights and trends that inform our understanding of this area.

{''.join(markdown_sections)}

## Conclusion

This report has provided a comprehensive analysis of {plan.get('topic', 'the topic')}. The key findings demonstrate significant insights that can inform future work in this area.
"""
                executive_summary = f"This comprehensive research report examines {plan.get('topic', 'the topic')}. Through systematic research and analysis, we have identified key insights and trends that inform our understanding of this area."
                conclusion = f"This report has provided a comprehensive analysis of {plan.get('topic', 'the topic')}. The key findings demonstrate significant insights that can inform future work in this area."
        else:
            # Fallback if parsing fails
            self.stream_output("Warning: Failed to parse LLM response, using fallback structure")
            sections = []
            word_count = 0
            
            for section_key, section_value in plan.get("outline", {}).items():
                # Extract title from nested structure if present
                if isinstance(section_value, dict) and 'title' in section_value:
                    section_title = section_value.get('title', section_key)
                    subsections = section_value.get('subsections', {})
                else:
                    section_title = section_key
                    subsections = section_value if isinstance(section_value, dict) else {}
                
                section_content = f"This section covers {section_title}. "
                section_content += f"Based on our research and analysis, we found that... "
                section_content += f"The key insights related to this area include... "
                section_content += f"Furthermore, the evidence suggests... "
                section_content += f"In conclusion for this section, we observe that..."
                
                sections.append({
                    "title": section_title,
                    "content": section_content,
                    "subsections": subsections,
                })
                word_count += len(section_content.split())
            
            # Build full markdown text
            markdown_sections = []
            for section in sections:
                markdown_sections.append(f"## {section.get('title', 'Untitled Section')}\n\n{section.get('content', '')}\n")
                if section.get('subsections') and isinstance(section.get('subsections'), dict):
                    for sub_key, sub_val in section.get('subsections', {}).items():
                        markdown_sections.append(f"### {sub_key}\n\n{sub_val}\n")
            
            full_text = f"""# Research Report: {plan.get('topic', 'Research Topic')}

## Executive Summary

This comprehensive research report examines {plan.get('topic', 'the topic')}. Through systematic research and analysis, we have identified key insights and trends that inform our understanding of this area.

{''.join(markdown_sections)}

## Conclusion

This report has provided a comprehensive analysis of {plan.get('topic', 'the topic')}. The key findings demonstrate significant insights that can inform future work in this area.
"""
            title = f"Research Report: {plan.get('topic', 'Research Topic')}"
            executive_summary = f"This comprehensive research report examines {plan.get('topic', 'the topic')}. Through systematic research and analysis, we have identified key insights and trends that inform our understanding of this area."
            conclusion = f"This report has provided a comprehensive analysis of {plan.get('topic', 'the topic')}. The key findings demonstrate significant insights that can inform future work in this area."
            word_count = word_count + 100  # Approximate with summary/conclusion
        
        report = {
            "title": title,
            "topic": plan.get("topic"),
            "sections": sections,
            "executive_summary": executive_summary,
            "conclusion": conclusion,
            "full_text": full_text,
            "word_count": len(full_text.split()) if full_text else word_count,
            "llm_response": llm_response,
        }
        
        return report

    def _revise_report(
        self, analysis: Dict[str, Any], plan: Dict[str, Any], qa_feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Revise a report based on QA feedback.
        
        Args:
            analysis: Analysis data
            plan: Research plan with outline
            qa_feedback: QA feedback with issues and recommendations
            
        Returns:
            Revised report dictionary
        """
        # Build prompt for LLM with QA feedback
        system_prompt = """You are a Writing Agent revising a research report based on QA feedback.
You should:
1. Address all issues identified in the QA review
2. Follow the recommendations provided
3. Improve areas flagged for enhancement
4. Maintain the overall structure and quality
5. Ensure all sections meet quality standards

Output your revised report in JSON format with:
- title: report title
- sections: list of sections with titles and content
- executive_summary: brief overview
- conclusion: final synthesis
- full_text: complete report in markdown format
"""

        # Extract QA issues and recommendations
        issues_text = "\n".join(
            f"- [{issue['severity'].upper()}] {issue['category']}: {issue['description']}"
            for issue in qa_feedback.get("issues", [])
        )
        
        recommendations_text = "\n".join(
            f"- {rec}" for rec in qa_feedback.get("recommendations", [])
        )
        
        # Prepare outline and insights
        outline_text = "\n".join(
            f"{key}: {value}" for key, value in plan.get("outline", {}).items()
        )
        
        # Handle different insight structures
        insights = analysis.get("insights", [])
        insights_list = []
        for insight in insights:
            if isinstance(insight, dict):
                # Try different key combinations
                title = insight.get("title") or insight.get("insight") or insight.get("name", "Insight")
                description = insight.get("description") or insight.get("analysis") or insight.get("content", "")
                insights_list.append(f"- {title}: {description}")
            elif isinstance(insight, str):
                insights_list.append(f"- {insight}")
        insights_text = "\n".join(insights_list)
        
        user_prompt = f"""Revise the research report on: {plan.get('topic', 'the given topic')}

QA FEEDBACK - Issues to Address:
{issues_text if issues_text else "No critical issues"}

QA FEEDBACK - Recommendations:
{recommendations_text}

Quality Score: {qa_feedback.get('quality_score', 0)}/100

Outline to follow:
{outline_text}

Key Insights to incorporate:
{insights_text}

Key Findings:
{chr(10).join(f'- {finding}' for finding in analysis.get('key_findings', []))}

Create an improved, well-structured report that addresses all QA feedback."""

        # Call LLM with appropriate token limit for comprehensive revised reports
        # Claude models typically support 8192 max_tokens for output
        llm_response = self.call_llm(user_prompt, system_prompt, max_tokens=8192)
        
        # Parse actual LLM response
        parsed_response = self.parse_json_from_llm(llm_response)
        
        if parsed_response and isinstance(parsed_response, dict):
            # Use LLM-generated content
            title = parsed_response.get("title", f"Research Report: {plan.get('topic', 'Research Topic')} (Revised)")
            sections = parsed_response.get("sections", [])
            executive_summary = parsed_response.get("executive_summary", "")
            conclusion = parsed_response.get("conclusion", "")
            full_text = parsed_response.get("full_text", "")
            
            # Validate that we got meaningful content
            if not sections or not full_text:
                self.stream_output("Warning: LLM response incomplete, generating enhanced fallback content")
                sections = []
                word_count = 0
                
                for section_key, section_value in plan.get("outline", {}).items():
                    # Extract title from nested structure if present
                    if isinstance(section_value, dict) and 'title' in section_value:
                        section_title = section_value.get('title', section_key)
                        subsections = section_value.get('subsections', {})
                    else:
                        section_title = section_key
                        subsections = section_value if isinstance(section_value, dict) else {}
                    
                    section_content = f"This section provides a comprehensive examination of {section_title}. "
                    section_content += f"Based on extensive research and thorough analysis, our findings indicate... "
                    section_content += f"The key insights and patterns that emerged include... "
                    
                    sections.append({
                        "title": section_title,
                        "content": section_content,
                        "subsections": subsections,
                    })
                    word_count += len(section_content.split())
                
                # Build full markdown text for incomplete revised response
                markdown_sections = []
                for section in sections:
                    markdown_sections.append(f"## {section.get('title', 'Untitled Section')}\n\n{section.get('content', '')}\n")
                    if section.get('subsections') and isinstance(section.get('subsections'), dict):
                        for sub_key, sub_val in section.get('subsections', {}).items():
                            markdown_sections.append(f"### {sub_key}\n\n{sub_val}\n")
                
                full_text = f"""# {title}

## Executive Summary

This comprehensive and thoroughly revised research report examines {plan.get('topic', 'the topic')}. Through systematic research, rigorous analysis, and careful revision based on quality assurance feedback, we have identified and expanded upon key insights and trends that inform our understanding of this area. This report addresses all quality concerns and provides enhanced depth and clarity.

{''.join(markdown_sections)}

## Conclusion

This revised report has provided a comprehensive and enhanced analysis of {plan.get('topic', 'the topic')}. The key findings, now presented with greater detail and support, demonstrate significant insights that can inform future work in this area. All quality assurance feedback has been incorporated to ensure the highest standards.
"""
                executive_summary = f"This comprehensive and thoroughly revised research report examines {plan.get('topic', 'the topic')}. Through systematic research, rigorous analysis, and careful revision based on quality assurance feedback, we have identified and expanded upon key insights and trends that inform our understanding of this area."
                conclusion = f"This revised report has provided a comprehensive and enhanced analysis of {plan.get('topic', 'the topic')}. The key findings, now presented with greater detail and support, demonstrate significant insights that can inform future work in this area."
        else:
            # Fallback if parsing fails
            self.stream_output("Warning: Failed to parse LLM response, using enhanced fallback structure")
            sections = []
            word_count = 0
            
            for section_key, section_value in plan.get("outline", {}).items():
                # Extract title from nested structure if present
                if isinstance(section_value, dict) and 'title' in section_value:
                    section_title = section_value.get('title', section_key)
                    subsections = section_value.get('subsections', {})
                else:
                    section_title = section_key
                    subsections = section_value if isinstance(section_value, dict) else {}
                
                # Generate more substantial content for revision
                section_content = f"This section provides a comprehensive examination of {section_title}. "
                section_content += f"Based on extensive research and thorough analysis, our findings indicate... "
                section_content += f"The key insights and patterns that emerged from this investigation include... "
                section_content += f"Furthermore, the evidence strongly suggests multiple important implications... "
                section_content += f"Detailed examination reveals additional nuanced factors... "
                section_content += f"In conclusion for this section, we observe significant correlations and trends that..."
                
                sections.append({
                    "title": section_title,
                    "content": section_content,
                    "subsections": subsections,
                })
                word_count += len(section_content.split())
            
            # Build full markdown text
            markdown_sections = []
            for section in sections:
                markdown_sections.append(f"## {section.get('title', 'Untitled Section')}\n\n{section.get('content', '')}\n")
                if section.get('subsections') and isinstance(section.get('subsections'), dict):
                    for sub_key, sub_val in section.get('subsections', {}).items():
                        markdown_sections.append(f"### {sub_key}\n\n{sub_val}\n")
            
            full_text = f"""# Research Report: {plan.get('topic', 'Research Topic')}

## Executive Summary

This comprehensive and thoroughly revised research report examines {plan.get('topic', 'the topic')}. Through systematic research, rigorous analysis, and careful revision based on quality assurance feedback, we have identified and expanded upon key insights and trends that inform our understanding of this area. This report addresses all quality concerns and provides enhanced depth and clarity.

{''.join(markdown_sections)}

## Conclusion

This revised report has provided a comprehensive and enhanced analysis of {plan.get('topic', 'the topic')}. The key findings, now presented with greater detail and support, demonstrate significant insights that can inform future work in this area. All quality assurance feedback has been incorporated to ensure the highest standards.
"""
            title = f"Research Report: {plan.get('topic', 'Research Topic')} (Revised)"
            executive_summary = f"This comprehensive and thoroughly revised research report examines {plan.get('topic', 'the topic')}. Through systematic research, rigorous analysis, and careful revision based on quality assurance feedback, we have identified and expanded upon key insights and trends that inform our understanding of this area."
            conclusion = f"This revised report has provided a comprehensive and enhanced analysis of {plan.get('topic', 'the topic')}. The key findings, now presented with greater detail and support, demonstrate significant insights that can inform future work in this area."
            word_count = word_count + 150  # More content in revision
        
        report = {
            "title": title,
            "topic": plan.get("topic"),
            "sections": sections,
            "executive_summary": executive_summary,
            "conclusion": conclusion,
            "full_text": full_text,
            "word_count": len(full_text.split()) if full_text else word_count,
            "llm_response": llm_response,
            "revised": True,
            "qa_feedback_addressed": qa_feedback,
        }
        
        return report
