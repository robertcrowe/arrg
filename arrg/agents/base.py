"""Base agent classes and interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional
import logging
import json
import re

from arrg.protocol import A2AMessage, MessageType, SharedWorkspace


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the ARRG system.
    """

    def __init__(
        self,
        agent_id: str,
        model: str,
        workspace: SharedWorkspace,
        api_key: str,
        provider_endpoint: str = "Tetrate",
        stream_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            model: Model string (e.g., 'gpt-4o', 'claude-3-5-sonnet')
            workspace: Shared workspace for storing artifacts
            api_key: API key for the model provider
            provider_endpoint: API provider endpoint
            stream_callback: Optional callback for streaming output
        """
        self.agent_id = agent_id
        self.model = model
        self.workspace = workspace
        self.api_key = api_key
        self.provider_endpoint = provider_endpoint
        self.stream_callback = stream_callback
        self.logger = logging.getLogger(f"arrg.agent.{agent_id}")
        
        # Message history for this agent
        self.message_history: list[A2AMessage] = []

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return the capabilities of this agent.
        
        Returns:
            Dictionary describing agent capabilities
        """
        pass

    @abstractmethod
    def process_message(self, message: A2AMessage) -> A2AMessage:
        """
        Process an incoming A2A message and return a response.
        
        Args:
            message: Incoming A2A message
            
        Returns:
            Response message
        """
        pass

    def send_message(self, message: A2AMessage):
        """
        Log an outgoing message.
        
        Args:
            message: Message being sent
        """
        self.message_history.append(message)
        self.logger.info(
            f"Sent {message.message_type.value} from {message.sender} to {message.receiver}"
        )

    def receive_message(self, message: A2AMessage):
        """
        Log an incoming message.
        
        Args:
            message: Message being received
        """
        self.message_history.append(message)
        self.logger.info(
            f"Received {message.message_type.value} from {message.sender}"
        )

    def stream_output(self, text: str):
        """
        Stream output to the dashboard console.
        
        Args:
            text: Text to stream
        """
        if self.stream_callback:
            self.stream_callback(f"[{self.agent_id}] {text}")
        self.logger.debug(text)

    def call_llm(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 4096) -> str:
        """
        Call the LLM with the given prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate (default: 4096)
            
        Returns:
            LLM response text
        """
        from arrg.utils.llm_client import LLMClient
        
        self.stream_output(f"Calling LLM ({self.model}) with prompt...")
        self.logger.info(f"LLM Call: {prompt[:100]}...")
        
        # Create LLM client and make the call
        try:
            client = LLMClient(
                provider=self.provider_endpoint,
                api_key=self.api_key,
                model=self.model,
            )
            response = client.call(prompt, system_prompt, max_tokens=max_tokens)
            return response
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return f"[Error: {str(e)}]"

    def create_task_complete_message(
        self,
        receiver: str,
        result: Dict[str, Any],
        in_reply_to: Optional[str] = None,
    ) -> A2AMessage:
        """
        Create a task completion message.
        
        Args:
            receiver: ID of the receiving agent
            result: Task result data
            in_reply_to: Optional message ID being replied to
            
        Returns:
            Task complete message
        """
        return A2AMessage(
            message_type=MessageType.TASK_COMPLETE,
            sender=self.agent_id,
            receiver=receiver,
            payload=result,
            in_reply_to=in_reply_to,
        )

    def create_error_message(
        self,
        receiver: str,
        error: str,
        in_reply_to: Optional[str] = None,
    ) -> A2AMessage:
        """
        Create an error message.
        
        Args:
            receiver: ID of the receiving agent
            error: Error description
            in_reply_to: Optional message ID being replied to
            
        Returns:
            Error message
        """
        return A2AMessage(
            message_type=MessageType.ERROR,
            sender=self.agent_id,
            receiver=receiver,
            payload={"error": error},
            in_reply_to=in_reply_to,
        )

    def parse_json_from_llm(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from LLM response, handling markdown code blocks and malformed JSON.
        
        Args:
            llm_response: Raw LLM response text
            
        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        if not llm_response or llm_response.startswith("[Error:"):
            self.logger.warning("LLM response is empty or error")
            return None
        
        # Try to extract JSON from markdown code blocks
        json_patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json ... ```
            r'```\s*\n(.*?)\n```',       # ``` ... ```
            r'\{.*\}',                    # Direct JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, llm_response, re.DOTALL)
            if matches:
                json_str = matches[0] if isinstance(matches[0], str) else matches[0]
                try:
                    parsed = json.loads(json_str)
                    self.logger.info("Successfully parsed JSON from LLM response")
                    return parsed
                except json.JSONDecodeError as e:
                    self.logger.debug(f"Failed to parse with pattern {pattern}: {e}")
                    continue
        
        # If no pattern worked, try parsing the entire response
        try:
            parsed = json.loads(llm_response)
            self.logger.info("Successfully parsed entire response as JSON")
            return parsed
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON from LLM response: {e}")
            return None
