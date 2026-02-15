"""Planning Agent - Creates research plans and outlines.

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


class PlanningAgent(BaseAgent):
    """
    Planning Agent creates research plans and outlines.
    Decomposes the research topic into structured research questions and sections.

    A2A Protocol: Receives a Task with topic in the user Message's DataPart,
    produces an Artifact containing the research plan.
    """

    def get_capabilities(self) -> Dict[str, Any]:
        """Return the capabilities of the Planning Agent."""
        return {
            "agent_type": "planning",
            "description": "Creates research plans and outlines",
            "capabilities": [
                "topic_decomposition",
                "outline_generation",
                "research_question_formulation",
                "section_planning",
            ],
            "inputs": ["topic", "requirements"],
            "outputs": ["research_plan", "outline", "research_questions"],
        }

    def process_task(self, task: Task, message: Message) -> Task:
        """
        Process an A2A Task to create a research plan.

        Args:
            task: A2A Task (in SUBMITTED state)
            message: User Message containing topic and requirements

        Returns:
            Updated Task with research plan Artifact (COMPLETED or FAILED)
        """
        self.receive_message(message)
        task.update_state(TaskState.WORKING, message="Creating research plan")
        task.add_to_history(message)

        try:
            # Extract topic from message DataPart
            data = message.get_data() or {}
            topic = data.get("topic", message.get_text())
            requirements = data.get("requirements", {})

            self.stream_output(f"Creating research plan for topic: {topic}")

            # Generate research plan
            plan = self._create_research_plan(topic, requirements)

            # Store plan in workspace (for cross-agent reference)
            plan_key = f"research_plan_{task.id}"
            self.workspace.store(plan_key, plan, persist=True)

            self.stream_output("Research plan created successfully")

            # Complete the task with result
            result = {
                "plan_reference": plan_key,
                "outline": plan["outline"],
                "research_questions": plan["research_questions"],
            }
            return self.create_completed_task(
                task, result_data=result,
                result_text="Research plan created successfully",
            )

        except Exception as e:
            self.stream_output(f"Error creating research plan: {str(e)}")
            return self.create_failed_task(task, error=str(e))

    def _create_research_plan(
        self, topic: str, requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a structured research plan for the given topic.

        Args:
            topic: Research topic
            requirements: Additional requirements

        Returns:
            Research plan dictionary
        """
        # Build prompt for LLM
        system_prompt = """You are a Planning Agent that creates comprehensive research plans.
Given a research topic, you must:
1. Break down the topic into key research questions
2. Create a structured outline with sections and subsections
3. Identify key areas that need investigation
4. Suggest research methodologies

Output your plan in JSON format with:
- research_questions: list of specific questions to answer
- outline: hierarchical structure with sections and subsections
- key_areas: main areas to investigate
- methodology: suggested research approaches
"""

        user_prompt = f"""Create a comprehensive research plan for the following topic:

Topic: {topic}

Requirements:
{requirements}

Provide a detailed research plan with research questions, outline, and methodology."""

        # Call LLM
        llm_response = self.call_llm(user_prompt, system_prompt)

        # Parse actual LLM response
        parsed_response = self.parse_json_from_llm(llm_response)

        if parsed_response and isinstance(parsed_response, dict):
            # Use LLM-generated content
            research_questions = parsed_response.get("research_questions", [])
            outline = parsed_response.get("outline", {})
            key_areas = parsed_response.get("key_areas", [])
            methodology = parsed_response.get("methodology", [])

            # Validate that we got meaningful content
            if not research_questions or not outline:
                self.stream_output("Warning: LLM response incomplete, using fallback structure")
                research_questions = [
                    f"What is the current state of {topic}?",
                    f"What are the key challenges in {topic}?",
                    f"What are the future trends in {topic}?",
                ]
                outline = {
                    "1. Introduction": "Background and context",
                    "2. Current State": "Overview and developments",
                    "3. Analysis": "Challenges and evaluation",
                    "4. Future Directions": "Trends and recommendations",
                    "5. Conclusion": "Summary of findings",
                }
        else:
            # Fallback if parsing fails
            self.stream_output("Warning: Failed to parse LLM response, using fallback structure")
            research_questions = [
                f"What is the current state of {topic}?",
                f"What are the key challenges in {topic}?",
                f"What are the future trends in {topic}?",
            ]
            outline = {
                "1. Introduction": {
                    "1.1": "Background and context",
                    "1.2": "Research objectives",
                },
                "2. Current State": {
                    "2.1": "Overview",
                    "2.2": "Key developments",
                },
                "3. Analysis": {
                    "3.1": "Challenges and opportunities",
                    "3.2": "Critical evaluation",
                },
                "4. Future Directions": {
                    "4.1": "Emerging trends",
                    "4.2": "Recommendations",
                },
                "5. Conclusion": {
                    "5.1": "Summary of findings",
                    "5.2": "Final thoughts",
                },
            }
            key_areas = [
                "Current state and background",
                "Technical challenges",
                "Market trends",
                "Future outlook",
            ]
            methodology = [
                "Literature review",
                "Data analysis",
                "Expert perspectives",
            ]

        plan = {
            "topic": topic,
            "research_questions": research_questions,
            "outline": outline,
            "key_areas": key_areas,
            "methodology": methodology,
            "llm_response": llm_response,
        }

        return plan
