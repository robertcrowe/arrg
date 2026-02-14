"""
Base agent classes and interfaces.

Supports A2A messaging with MCP (Model Context Protocol) 2025-11-25 tool integration.
All tool-calling flows through the MCP protocol using tools/list and tools/call.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, List
import logging
import json
import re

from arrg.protocol import A2AMessage, MessageType, SharedWorkspace
from arrg.a2a import AgentCard, AgentCapability
from arrg.mcp import MCPToolRegistry, MCPToolCall, MCPToolResult, TextContent, get_tool_registry


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
        
        # Initialize MCP tool registry
        self.tool_registry = get_tool_registry()
        
        # Create AgentCard for A2A protocol
        capabilities = self.get_capabilities()
        
        # Map agent type to AgentCapability enum
        agent_type_map = {
            "planning": AgentCapability.PLANNING,
            "research": AgentCapability.RESEARCH,
            "analysis": AgentCapability.ANALYSIS,
            "writing": AgentCapability.WRITING,
            "qa": AgentCapability.QA,
        }
        
        # Get the primary capability based on agent_type
        agent_type = capabilities.get("agent_type", agent_id)
        primary_capability = agent_type_map.get(agent_type, AgentCapability.DATA_PROCESSING)
        
        self.agent_card = AgentCard(
            agent_id=agent_id,
            name=agent_id.title() + " Agent",
            description=capabilities.get("description", f"Agent for {agent_id}"),
            capabilities=[primary_capability],
            supported_message_types=["task", "query", "response"],
            metadata=capabilities,
            version="1.0.0"
        )

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

    def call_llm(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        max_tokens: int = 8192,
        use_tools: bool = False,
        max_tool_rounds: int = 5,
    ) -> str:
        """
        Call the LLM with the given prompt, optionally with MCP tools.
        
        When use_tools=True, this implements an agentic tool-call execution loop:
        1. Send prompt + MCP tool schemas (via tools/list) to the LLM
        2. If LLM responds with tool_calls, execute each via MCP tools/call
        3. Feed MCP tool results back to the LLM as tool messages
        4. Repeat until LLM responds with text content (no more tool_calls)
        
        All tool definitions come from MCPToolRegistry.get_tools_for_llm()
        (MCP tools/list → OpenAI format bridge).  All tool execution goes
        through MCPToolRegistry.call_tool() (MCP tools/call).
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate (default: 8192)
            use_tools: Whether to include MCP tools in the call
            max_tool_rounds: Maximum rounds of tool-call → result loops (default: 5)
            
        Returns:
            LLM response text (final text after all tool calls are resolved)
        """
        from arrg.utils.llm_client import LLMClient
        
        tools_info = " with MCP tools" if use_tools else ""
        self.stream_output(f"Calling LLM ({self.model}){tools_info} max_tokens={max_tokens}...")
        self.logger.info(f"LLM Call with max_tokens={max_tokens}{tools_info}: {prompt[:100]}...")
        
        try:
            client = LLMClient(
                provider=self.provider_endpoint,
                api_key=self.api_key,
                model=self.model,
            )
            
            # Get MCP tool schemas for LLM (MCP tools/list → OpenAI format)
            tools = None
            if use_tools:
                tools = self.tool_registry.get_tools_for_llm()
                tool_names = [t['function']['name'] for t in tools]
                self.logger.info(f"MCP tools/list returned {len(tools)} tools: {tool_names}")
            
            if not use_tools:
                # Simple call without tools
                response = client.call(
                    prompt, 
                    system_prompt, 
                    max_tokens=max_tokens,
                    tools=None,
                )
                return response
            
            # --- Agentic tool-call execution loop ---
            # Build conversation messages for multi-turn tool use
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            for round_num in range(max_tool_rounds):
                # Call LLM with tools and full message history
                response = client.call_with_messages(
                    messages=messages,
                    max_tokens=max_tokens,
                    tools=tools,
                )
                
                # Check if LLM returned tool calls
                if not response.get("tool_calls"):
                    # No tool calls — LLM is done, return text content
                    text = response.get("content", "")
                    self.logger.info(f"LLM finished after {round_num} tool-call round(s)")
                    return text
                
                # LLM wants to call tools — execute each via MCP tools/call
                tool_calls = response["tool_calls"]
                assistant_content = response.get("content", None)
                
                # Add the assistant's message (with tool_calls) to history
                assistant_msg: Dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
                if assistant_content:
                    assistant_msg["content"] = assistant_content
                messages.append(assistant_msg)
                
                self.stream_output(f"LLM requested {len(tool_calls)} MCP tool call(s) (round {round_num + 1})")
                
                for tc in tool_calls:
                    tc_id = tc.get("id", f"call_{round_num}")
                    func = tc.get("function", {})
                    tool_name = func.get("name", "unknown")
                    
                    # Parse arguments (LLM may return as JSON string)
                    raw_args = func.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            tool_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            tool_args = {}
                    else:
                        tool_args = raw_args
                    
                    self.logger.info(f"MCP tools/call: {tool_name}({tool_args})")
                    self.stream_output(f"  → MCP tools/call: {tool_name}")
                    
                    # Execute via MCP tools/call
                    mcp_call = MCPToolCall(
                        name=tool_name,
                        arguments=tool_args,
                        call_id=tc_id,
                    )
                    mcp_result: MCPToolResult = self.tool_registry.call_tool(mcp_call)
                    
                    result_text = mcp_result.get_text()
                    if mcp_result.is_error:
                        self.stream_output(f"  ✗ Tool error: {result_text[:100]}")
                    else:
                        self.stream_output(f"  ✓ Tool returned {len(result_text)} chars")
                    
                    # Add MCP tool result to conversation as a tool message
                    # (bridges MCP result back into LLM conversation format)
                    messages.append(mcp_result.to_llm_tool_result())
            
            # Exhausted tool rounds — make one final call without tools
            self.logger.warning(f"Exhausted {max_tool_rounds} tool-call rounds, making final call without tools")
            self.stream_output(f"Tool loop limit reached ({max_tool_rounds} rounds), getting final response...")
            response = client.call_with_messages(
                messages=messages,
                max_tokens=max_tokens,
                tools=None,
            )
            return response.get("content", "")
            
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
        
        # Log response length for debugging
        self.logger.debug(f"LLM response length: {len(llm_response)} chars")
        
        # CHECK FOR TRUNCATION FIRST - before trying to extract nested objects
        # Look for signs of truncation: incomplete strings, unclosed structures, etc.
        is_truncated = any([
            llm_response.rstrip().endswith(('",', '",\n', '"', ',')),
            llm_response.count('{') > llm_response.count('}'),
            llm_response.count('[') > llm_response.count(']'),
            llm_response.count('"') % 2 != 0,  # Odd number of quotes
        ])
        
        if is_truncated:
            self.logger.warning("Response appears truncated - attempting repair FIRST")
            repaired = self._attempt_json_repair(llm_response)
            if repaired:
                self.logger.info("Successfully repaired and parsed truncated JSON")
                return repaired
        
        # Try to extract JSON from markdown code blocks
        # Look for ```json or ``` code fences
        code_fence_pattern = r'```(?:json)?\s*\n(.*?)(?:\n```|$)'
        code_matches = re.findall(code_fence_pattern, llm_response, re.DOTALL)
        
        if code_matches:
            self.logger.debug(f"Found {len(code_matches)} code fence blocks")
            for i, json_str in enumerate(code_matches):
                json_str = json_str.strip()
                if not json_str:
                    continue
                    
                self.logger.debug(f"Attempting to parse code fence block {i+1}")
                parsed = self._try_parse_json(json_str)
                if parsed:
                    self.logger.info(f"Successfully parsed JSON from code fence block {i+1}")
                    return parsed
        
        # Try to find JSON object in the response (look for outermost braces)
        # Use non-greedy match to find first complete JSON object
        json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_object_pattern, llm_response, re.DOTALL)
        
        if json_matches:
            self.logger.debug(f"Found {len(json_matches)} potential JSON objects")
            # Try the longest match first (likely to be the complete JSON)
            for json_str in sorted(json_matches, key=len, reverse=True):
                parsed = self._try_parse_json(json_str)
                if parsed:
                    self.logger.info("Successfully parsed JSON object from response")
                    return parsed
        
        # If no pattern worked, try parsing the entire response
        parsed = self._try_parse_json(llm_response)
        if parsed:
            self.logger.info("Successfully parsed entire response as JSON")
            return parsed
        
        self.logger.warning(f"Failed to parse JSON from LLM response (tried all methods)")
        self.logger.debug(f"Response preview: {llm_response[:500]}...")
        return None
    
    def _try_parse_json(self, json_str: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse a JSON string, with basic error handling.
        
        Args:
            json_str: String that might contain JSON
            
        Returns:
            Parsed dictionary or None if parsing fails
        """
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed
            else:
                self.logger.debug(f"Parsed JSON is not a dict: {type(parsed)}")
                return None
        except json.JSONDecodeError as e:
            self.logger.debug(f"JSON parse error: {e}")
            return None
    
    def _attempt_json_repair(self, json_str: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair truncated JSON by closing open structures.
        
        Args:
            json_str: Potentially truncated JSON string
            
        Returns:
            Parsed dictionary or None if repair fails
        """
        original_str = json_str
        
        # Try to extract JSON from markdown code fences first
        code_fence_match = re.search(r'```(?:json)?\s*\n(.*?)(?:\n```|$)', json_str, re.DOTALL)
        if code_fence_match:
            json_str = code_fence_match.group(1).strip()
        else:
            # Remove markdown code fences if present
            json_str = re.sub(r'^```(?:json)?\s*\n', '', json_str)
            json_str = re.sub(r'\n```\s*$', '', json_str)
            json_str = json_str.strip()
        
        # Count unclosed braces and brackets (but ignore those inside strings)
        # Properly track escape sequences and find safe truncation points
        in_string = False
        escape_next = False
        open_braces = 0
        open_brackets = 0
        depth_stack = []  # Track { and [ positions
        last_comma_at_depth = {}  # Track last comma at each depth level
        current_depth = 0
        
        for idx, char in enumerate(json_str):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    open_braces += 1
                    current_depth += 1
                    depth_stack.append(('brace', idx))
                elif char == '}':
                    open_braces -= 1
                    current_depth -= 1
                    if depth_stack and depth_stack[-1][0] == 'brace':
                        depth_stack.pop()
                elif char == '[':
                    open_brackets += 1
                    current_depth += 1
                    depth_stack.append(('bracket', idx))
                elif char == ']':
                    open_brackets -= 1
                    current_depth -= 1
                    if depth_stack and depth_stack[-1][0] == 'bracket':
                        depth_stack.pop()
                elif char == ',':
                    # Track commas at each depth - these are safe truncation points
                    last_comma_at_depth[current_depth] = idx
        
        if open_braces > 0 or open_brackets > 0 or in_string:
            self.logger.debug(f"Attempting repair: {open_braces} unclosed braces, {open_brackets} unclosed brackets, in_string={in_string}")
            
            # Strategy 1: Try to find the last comma and truncate there
            # This gives us the last complete key-value pair in an object/array
            best_truncation = -1
            if last_comma_at_depth:
                # Find the deepest comma (most specific truncation point)
                for depth in sorted(last_comma_at_depth.keys(), reverse=True):
                    best_truncation = last_comma_at_depth[depth]
                    self.logger.debug(f"Found comma at depth {depth}, index {best_truncation}")
                    break
            
            if best_truncation > 0:
                # Try truncating before the last comma
                truncated = json_str[:best_truncation].rstrip()
                self.logger.debug(f"Truncating at last comma: '{truncated[-50:]}'...")
                
                # Recount structures in truncated string
                in_str = False
                esc_next = False
                o_braces = 0
                o_brackets = 0
                
                for char in truncated:
                    if esc_next:
                        esc_next = False
                        continue
                    if char == '\\':
                        esc_next = True
                        continue
                    if char == '"':
                        in_str = not in_str
                        continue
                    if not in_str:
                        if char == '{':
                            o_braces += 1
                        elif char == '}':
                            o_braces -= 1
                        elif char == '[':
                            o_brackets += 1
                        elif char == ']':
                            o_brackets -= 1
                
                # Close the truncated structures
                repaired = truncated
                if in_str:
                    repaired += '"'
                repaired += '}' * o_braces
                repaired += ']' * o_brackets
                
                parsed = self._try_parse_json(repaired)
                if parsed:
                    self.logger.info("Successfully repaired JSON by truncating at last comma")
                    return parsed
            
            # Strategy 2: Simple closure - just close what's open
            repaired = json_str
            
            # If we're inside a string, close it
            if in_string:
                repaired += '"'
            
            # Remove trailing commas before closing structures
            repaired = re.sub(r',\s*$', '', repaired)
            
            # Close open structures in the correct order
            repaired += '}' * open_braces
            repaired += ']' * open_brackets
            
            # Try to parse the repaired JSON
            parsed = self._try_parse_json(repaired)
            if parsed:
                self.logger.info("Successfully repaired JSON by closing open structures")
                return parsed
            
            # Strategy 3: Remove incomplete lines progressively
            lines = json_str.split('\n')
            if len(lines) > 1:
                # Try removing progressively more lines until we get valid JSON
                for i in range(len(lines) - 1, max(0, len(lines) - 5), -1):
                    truncated = '\n'.join(lines[:i])
                    # Recount structures
                    in_string = False
                    escape_next = False
                    open_braces = 0
                    open_brackets = 0
                    
                    for char in truncated:
                        if escape_next:
                            escape_next = False
                            continue
                        if char == '\\':
                            escape_next = True
                            continue
                        if char == '"':
                            in_string = not in_string
                            continue
                        if not in_string:
                            if char == '{':
                                open_braces += 1
                            elif char == '}':
                                open_braces -= 1
                            elif char == '[':
                                open_brackets += 1
                            elif char == ']':
                                open_brackets -= 1
                    
                    # Close the structures
                    repaired = truncated
                    if in_string:
                        repaired += '"'
                    repaired = re.sub(r',\s*$', '', repaired)
                    repaired += ']' * open_brackets
                    repaired += '}' * open_braces
                    
                    parsed = self._try_parse_json(repaired)
                    if parsed:
                        self.logger.info(f"Successfully repaired by removing {len(lines) - i} incomplete lines")
                        return parsed
        
        return None
