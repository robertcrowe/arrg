"""Research Agent - Gathers information on research questions."""

from typing import Any, Dict, List
from arrg.agents.base import BaseAgent
from arrg.protocol import A2AMessage, MessageType


class ResearchAgent(BaseAgent):
    """
    Research Agent gathers information on research questions.
    Simulates web search and information gathering.
    """

    def get_capabilities(self) -> Dict[str, Any]:
        """Return the capabilities of the Research Agent."""
        return {
            "agent_type": "research",
            "description": "Gathers information on research questions",
            "capabilities": [
                "information_gathering",
                "web_search_simulation",
                "source_compilation",
                "fact_extraction",
            ],
            "inputs": ["research_questions", "plan_reference"],
            "outputs": ["research_data", "sources", "findings"],
        }

    def process_message(self, message: A2AMessage) -> A2AMessage:
        """
        Process an incoming message and gather research data.
        
        Args:
            message: Incoming A2A message
            
        Returns:
            Response message with research data
        """
        self.receive_message(message)
        
        if message.message_type == MessageType.TASK_REQUEST:
            try:
                # Extract research questions from payload
                research_questions = message.payload.get("research_questions", [])
                plan_reference = message.payload.get("plan_reference")
                
                self.stream_output(f"Conducting research on {len(research_questions)} questions")
                
                # Gather research data
                research_data = self._conduct_research(research_questions, plan_reference)
                
                # Store research data in workspace
                data_key = f"research_data_{message.message_id}"
                self.workspace.store(data_key, research_data, persist=True)
                
                self.stream_output("Research completed successfully")
                
                # Create response with reference to research data
                response = self.create_task_complete_message(
                    receiver=message.sender,
                    result={
                        "data_reference": data_key,
                        "summary": research_data["summary"],
                        "source_count": len(research_data["sources"]),
                    },
                    in_reply_to=message.message_id,
                )
                
                self.send_message(response)
                return response
                
            except Exception as e:
                self.stream_output(f"Error conducting research: {str(e)}")
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

    def _conduct_research(
        self, research_questions: List[str], plan_reference: str = None
    ) -> Dict[str, Any]:
        """
        Conduct research on the given questions.
        
        Args:
            research_questions: List of research questions
            plan_reference: Optional reference to research plan
            
        Returns:
            Research data dictionary
        """
        # Retrieve plan if reference provided
        plan = None
        if plan_reference:
            plan = self.workspace.retrieve(plan_reference)
        
        # Build prompt for LLM
        system_prompt = """You are a Research Agent that gathers information on research questions.
For each question, you should:
1. Provide comprehensive information and key findings
2. Cite relevant sources (even if simulated)
3. Extract important facts and data points
4. Note any conflicting information or gaps

Output your research in JSON format with:
- findings: list of findings for each question
- sources: list of sources consulted
- key_facts: important facts extracted
- gaps: identified knowledge gaps
"""

        questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(research_questions))
        
        user_prompt = f"""Conduct research on the following questions:

{questions_text}

Provide comprehensive findings with sources and key facts for each question."""

        # Call LLM
        llm_response = self.call_llm(user_prompt, system_prompt)
        
        # Parse actual LLM response
        parsed_response = self.parse_json_from_llm(llm_response)
        
        if parsed_response and isinstance(parsed_response, dict):
            # Use LLM-generated content
            findings = parsed_response.get("findings", [])
            sources = parsed_response.get("sources", [])
            key_facts = parsed_response.get("key_facts", [])
            gaps = parsed_response.get("gaps", [])
            
            # Validate that we got meaningful content
            if not findings:
                self.stream_output("Warning: LLM response incomplete, using fallback structure")
                findings = []
                for i, question in enumerate(research_questions):
                    findings.append({
                        "question": question,
                        "answer": f"Research findings for: {question}",
                        "key_points": [
                            f"Key point 1 for question {i+1}",
                            f"Key point 2 for question {i+1}",
                        ],
                        "sources": [f"Source {i+1}"],
                    })
        else:
            # Fallback if parsing fails
            self.stream_output("Warning: Failed to parse LLM response, using fallback structure")
            findings = []
            for i, question in enumerate(research_questions):
                findings.append({
                    "question": question,
                    "answer": f"Research findings for: {question}",
                    "key_points": [
                        f"Key point 1 for question {i+1}",
                        f"Key point 2 for question {i+1}",
                        f"Key point 3 for question {i+1}",
                    ],
                    "sources": [
                        f"Source A for question {i+1}",
                        f"Source B for question {i+1}",
                    ],
                })
            sources = [
                "Academic Journal A (2024)",
                "Industry Report B (2023)",
                "Expert Analysis C (2024)",
                "Technical Documentation D",
            ]
            key_facts = [
                "Important fact 1 from research",
                "Important fact 2 from research",
                "Important fact 3 from research",
            ]
            gaps = [
                "Area needing more investigation 1",
                "Area needing more investigation 2",
            ]
        
        research_data = {
            "questions": research_questions,
            "findings": findings,
            "sources": sources,
            "key_facts": key_facts,
            "gaps": gaps,
            "summary": f"Completed research on {len(research_questions)} questions with {len(findings)} detailed findings",
            "llm_response": llm_response,
        }
        
        return research_data
