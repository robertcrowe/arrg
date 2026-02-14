"""
LLM Client for making calls to various providers.

Supports tool-calling via MCP (Model Context Protocol) 2025-11-25:
- call() returns plain text (simple prompts, no tool inspection)
- call_with_messages() returns structured responses including tool_calls
  for the agentic tool-call execution loop in BaseAgent.call_llm()
"""

import os
from typing import Optional, Dict, Any, List
import logging
import json


class LLMClient:
    """
    Client for making LLM calls to various providers.
    Supports OpenAI, Anthropic, and compatible APIs (like Tetrate).
    
    Tool integration follows MCP 2025-11-25: tool schemas are passed as
    OpenAI function-calling format (bridged from MCP tools/list), and
    tool_calls in responses trigger MCP tools/call execution in the agent.
    """

    def __init__(self, provider: str, api_key: str, model: str):
        """
        Initialize the LLM client.
        
        Args:
            provider: Provider name (Tetrate, OpenAI, Anthropic, Local)
            api_key: API key for authentication
            model: Model identifier
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(f"arrg.llm_client.{provider}")
        
        # Initialize provider-specific client
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the provider-specific client."""
        try:
            if self.provider in ["OpenAI", "Tetrate"]:
                # OpenAI-compatible API
                from openai import OpenAI
                
                if self.provider == "Tetrate":
                    # Tetrate uses OpenAI-compatible API
                    # Note: Tetrate Agent Router Service requires the .router. subdomain
                    base_url = os.environ.get("TETRATE_API_BASE", "https://api.router.tetrate.ai/v1")
                    
                    # Tetrate requires custom headers for proper routing
                    default_headers = {
                        "HTTP-Referer": "https://github.com/yourusername/arrg",  # Identifies the application
                        "X-Title": "arrg"  # Application name
                    }
                    
                    self._client = OpenAI(
                        api_key=self.api_key,
                        base_url=base_url,
                        default_headers=default_headers,
                    )
                else:
                    self._client = OpenAI(api_key=self.api_key)
                    
            elif self.provider == "Anthropic":
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
                
            elif self.provider == "Local":
                # Local models via OpenAI-compatible API (e.g., Ollama, vLLM)
                from openai import OpenAI
                base_url = os.environ.get("LOCAL_API_BASE", "http://localhost:11434/v1")
                self._client = OpenAI(
                    api_key="local",  # Local doesn't need real key
                    base_url=base_url,
                )
            else:
                self.logger.warning(f"Unknown provider: {self.provider}, using mock mode")
                
        except ImportError as e:
            self.logger.warning(f"Failed to import provider SDK: {e}. Using mock mode.")
            self._client = None

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Make an LLM call with optional MCP tool support.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            tools: Optional MCP tools for function calling
            
        Returns:
            LLM response text
        """
        if not self._client:
            return self._mock_call(prompt, system_prompt)
        
        try:
            if self.provider in ["OpenAI", "Tetrate", "Local"]:
                return self._call_openai(prompt, system_prompt, temperature, max_tokens, tools)
            elif self.provider == "Anthropic":
                return self._call_anthropic(prompt, system_prompt, temperature, max_tokens, tools)
            else:
                return self._mock_call(prompt, system_prompt)
                
        except Exception as e:
            # Log the full error with stack trace for debugging
            self.logger.error(f"LLM call failed with error: {e}", exc_info=True)
            
            # Check if this is a Tetrate error-in-200 or client error that should propagate
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in [
                'tetrate service error',  # Tetrate 503/500 errors
                'tetrate api error',       # Generic Tetrate errors
                'context length exceeded', # Context length errors
                'authentication error',    # Auth errors
                'rate limit exceeded',     # Rate limit errors
                '400', 'bad request', 'invalid', 'max_tokens', 'context length'
            ]):
                # Don't fall back to mock for these errors - they need to be handled by caller
                self.logger.error(
                    f"API error detected (not falling back to mock): {e}\n"
                    f"Request details: max_tokens={max_tokens}, model={self.model}, provider={self.provider}\n"
                    f"This error should be handled by the calling code."
                )
                raise
            
            # For network/server errors only, fall back to mock mode with warning
            self.logger.warning(
                f"Network or server error, falling back to mock mode. "
                f"Original error: {e}"
            )
            return self._mock_call(prompt, system_prompt)

    def call_with_messages(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 8192,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Call the LLM with a full message history, returning structured output.
        
        This is the core method for the MCP tool-call execution loop.
        Returns a dict with 'content' (text) and optionally 'tool_calls'
        (list of tool call dicts in OpenAI format).
        
        When 'tool_calls' is present, the caller (BaseAgent.call_llm) should
        execute each via MCP tools/call and append tool results as tool
        messages before calling again.
        
        Args:
            messages: Full conversation history (system, user, assistant, tool messages)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional MCP tools in OpenAI function-calling format
            
        Returns:
            Dict with keys:
              - content: str (text response, may be empty if only tool_calls)
              - tool_calls: Optional[List[Dict]] (tool calls in OpenAI format)
        """
        if not self._client:
            return self._mock_call_with_messages(messages, tools)
        
        try:
            if self.provider in ["OpenAI", "Tetrate", "Local"]:
                return self._call_openai_with_messages(messages, temperature, max_tokens, tools)
            elif self.provider == "Anthropic":
                return self._call_anthropic_with_messages(messages, temperature, max_tokens, tools)
            else:
                return self._mock_call_with_messages(messages, tools)
        except Exception as e:
            self.logger.error(f"call_with_messages failed: {e}", exc_info=True)
            error_str = str(e).lower()
            if any(kw in error_str for kw in [
                'tetrate service error', 'tetrate api error',
                'context length exceeded', 'authentication error',
                'rate limit exceeded', '400', 'bad request', 'invalid',
                'max_tokens', 'context length',
            ]):
                raise
            self.logger.warning(f"Falling back to mock: {e}")
            return self._mock_call_with_messages(messages, tools)

    def _call_openai_with_messages(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Call OpenAI-compatible API with full message history.
        Returns structured response with content and optional tool_calls.
        """
        api_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if tools:
            api_kwargs["tools"] = tools
            api_kwargs["tool_choice"] = "auto"
        
        response = self._client.chat.completions.create(**api_kwargs)
        
        # Tetrate error-in-200 check
        if self.provider == "Tetrate":
            self._check_tetrate_error(response)
        
        # Validate response structure
        if isinstance(response, str):
            return {"content": response, "tool_calls": None}
        
        if not hasattr(response, 'choices') or not response.choices:
            raise ValueError(f"Invalid response from {self.provider}: no choices")
        
        message = response.choices[0].message
        content = message.content or ""
        
        # Check for tool calls
        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
        
        return {"content": content, "tool_calls": tool_calls}

    def _call_anthropic_with_messages(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Call Anthropic API with full message history.
        Returns structured response with content and optional tool_calls.
        """
        # Separate system prompt from messages
        system = None
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            elif msg["role"] == "tool":
                # Anthropic expects tool results in a specific format
                api_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": msg.get("content", ""),
                        }
                    ],
                })
            else:
                api_messages.append(msg)
        
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            # Convert OpenAI tool format to Anthropic format
            anthropic_tools = []
            for tool in tools:
                func = tool.get("function", {})
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
            kwargs["tools"] = anthropic_tools
        
        response = self._client.messages.create(**kwargs)
        
        content = ""
        tool_calls = None
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })
        
        return {"content": content, "tool_calls": tool_calls}

    def _mock_call_with_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Mock call_with_messages for testing without a real API.
        Returns text content (no tool_calls) based on the last user message.
        """
        self.logger.info("Using mock call_with_messages response")
        
        # Find the last user message for context
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
        
        # Find system prompt
        system = None
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
                break
        
        text = self._mock_call(last_user, system)
        return {"content": text, "tool_calls": None}

    def _check_tetrate_error(self, response: Any) -> None:
        """Check for Tetrate error-in-200-response pattern."""
        error_obj = None
        if hasattr(response, 'error'):
            error_obj = response.error
        elif isinstance(response, dict) and 'error' in response:
            error_obj = response['error']
        
        if error_obj:
            if isinstance(error_obj, dict):
                error_message = error_obj.get('message', 'Unknown Tetrate error')
                error_code = error_obj.get('code', 0)
            else:
                error_message = str(error_obj)
                error_code = 0
            
            self.logger.error(f"Tetrate error in 200: [{error_code}] {error_message}")
            
            if error_code == 400 and 'maximum context length' in error_message.lower():
                raise ValueError(f"Context length exceeded: {error_message}")
            elif error_code in [401, 403]:
                raise ValueError(f"Authentication error: {error_message}")
            elif error_code == 429:
                raise ValueError(f"Rate limit exceeded: {error_message}")
            elif error_code in [500, 503]:
                raise ValueError(f"Tetrate service error: {error_message}")
            else:
                raise ValueError(f"Tetrate API error [{error_code}]: {error_message}")

    def _call_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Make a call to OpenAI-compatible API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Build API call kwargs
            api_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Add tools if provided
            if tools:
                api_kwargs["tools"] = tools
                api_kwargs["tool_choice"] = "auto"
            
            response = self._client.chat.completions.create(**api_kwargs)
            
            # Debug logging to understand response structure
            self.logger.debug(f"Response type: {type(response)}")
            self.logger.debug(f"Response: {response}")
            
            # CRITICAL: Tetrate can return errors in HTTP 200 responses
            # Check for error object in the response (Tetrate-specific behavior)
            if self.provider == "Tetrate":
                # Check if response has an error attribute or is a dict with error
                error_obj = None
                if hasattr(response, 'error'):
                    error_obj = response.error
                elif isinstance(response, dict) and 'error' in response:
                    error_obj = response['error']
                
                if error_obj:
                    # Extract error details
                    if isinstance(error_obj, dict):
                        error_message = error_obj.get('message', 'Unknown Tetrate error')
                        error_code = error_obj.get('code', 0)
                    else:
                        error_message = str(error_obj)
                        error_code = 0
                    
                    self.logger.error(f"Tetrate returned error in 200 response: [{error_code}] {error_message}")
                    
                    # Handle specific error codes
                    if error_code == 400 and 'maximum context length' in error_message.lower():
                        raise ValueError(f"Context length exceeded: {error_message}")
                    elif error_code in [401, 403]:
                        raise ValueError(f"Authentication error: {error_message}")
                    elif error_code == 429:
                        raise ValueError(f"Rate limit exceeded: {error_message}")
                    elif error_code in [500, 503]:
                        raise ValueError(f"Tetrate service error: {error_message}")
                    else:
                        raise ValueError(f"Tetrate API error [{error_code}]: {error_message}")
            
            # Handle various response formats
            if isinstance(response, str):
                # Handle raw string responses (malformed API responses)
                if not response or response.strip() == "":
                    error_msg = f"Received empty string response from {self.provider} API"
                    if self.provider == "Tetrate":
                        error_msg += (
                            "\n\nTetrate API returned HTTP 200 with empty body. This indicates:\n"
                            "1. The Tetrate service may be down or misconfigured\n"
                            "2. The API endpoint might be incorrect\n"
                            "3. The API key may not be authorized\n\n"
                            "Recommendation: Try a different provider (OpenAI or Anthropic) or use mock mode.\n"
                            "See TROUBLESHOOTING.md for details."
                        )
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                self.logger.warning("Received raw string response instead of structured object")
                return response
            
            # Check if response has the expected structure
            if not hasattr(response, 'choices'):
                error_msg = f"Response missing 'choices' attribute. Type: {type(response)}, Content: {response}"
                if self.provider == "Tetrate":
                    error_msg += (
                        "\n\nTetrate API response is malformed (no 'choices' field). This indicates:\n"
                        "1. The API returned empty or invalid JSON\n"
                        "2. The service may be experiencing issues\n"
                        "3. The endpoint may not be fully operational\n\n"
                        "Recommendation: Try a different provider or check Tetrate service status.\n"
                        "See TROUBLESHOOTING.md for details."
                    )
                self.logger.error(error_msg)
                raise AttributeError(error_msg)
            
            if not response.choices or len(response.choices) == 0:
                self.logger.error("Response has empty choices array")
                raise ValueError("No choices in API response")
            
            # Extract content from the first choice
            content = response.choices[0].message.content
            
            if not content:
                self.logger.warning("Response content is empty")
                return ""
            
            return content
            
        except AttributeError as e:
            self.logger.error(f"AttributeError parsing response: {e}")
            # Provide helpful context for Tetrate-specific issues
            if self.provider == "Tetrate" and "'choices'" in str(e):
                raise ValueError(
                    f"Tetrate API is not responding correctly (empty response body).\n"
                    f"The API returned HTTP 200 but with no content.\n\n"
                    f"This is a known issue with the Tetrate service. Please:\n"
                    f"1. Switch to OpenAI or Anthropic provider in the dashboard\n"
                    f"2. Or use mock mode for testing\n"
                    f"3. Or contact Tetrate support about API availability\n\n"
                    f"Original error: {e}"
                )
            raise ValueError(f"Invalid API response structure from {self.provider}: {e}")
        except Exception as e:
            self.logger.error(f"Error in _call_openai: {e}", exc_info=True)
            raise

    def _call_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Make a call to Anthropic API."""
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        # Add tools if provided (Anthropic uses same format as OpenAI)
        if tools:
            kwargs["tools"] = tools
        
        response = self._client.messages.create(**kwargs)
        
        return response.content[0].text

    def _mock_call(self, prompt: str, system_prompt: Optional[str]) -> str:
        """
        Generate a mock response when no real client is available.
        This allows the system to work for testing without actual API calls.
        """
        self.logger.info("Using mock LLM response")
        
        # Generate contextual mock based on prompt keywords
        if "research plan" in prompt.lower() or "planning" in prompt.lower():
            return """
{
  "research_questions": [
    "What is the current state and recent developments?",
    "What are the key challenges and opportunities?",
    "What are the future trends and implications?"
  ],
  "outline": {
    "1. Introduction": {
      "1.1": "Background and context",
      "1.2": "Research objectives"
    },
    "2. Current State": {
      "2.1": "Overview",
      "2.2": "Key developments"
    },
    "3. Analysis": {
      "3.1": "Challenges and opportunities",
      "3.2": "Critical evaluation"
    },
    "4. Future Directions": {
      "4.1": "Emerging trends",
      "4.2": "Recommendations"
    },
    "5. Conclusion": {
      "5.1": "Summary",
      "5.2": "Final thoughts"
    }
  },
  "methodology": ["Literature review", "Data analysis", "Expert perspectives"]
}
"""
        elif "research" in prompt.lower() or "search" in prompt.lower():
            return """
[Mock research data gathered from multiple sources]

Key findings:
- Recent developments indicate significant progress
- Multiple stakeholders are involved
- Technology continues to evolve
- Market trends show growing interest
- Expert opinions vary on implementation approaches

Sources consulted:
- Academic publications
- Industry reports
- News articles
- Expert interviews
"""
        elif "analysis" in prompt.lower() or "synthesize" in prompt.lower():
            return """
Based on the research data, several key insights emerge:

1. **Cross-cutting themes**: Multiple sources highlight similar patterns
2. **Contradictions**: Some findings present differing perspectives
3. **Gaps**: Areas requiring further investigation
4. **Implications**: Potential impact on stakeholders
5. **Recommendations**: Suggested actions based on evidence

The analysis reveals both opportunities and challenges, with consensus
on certain core issues while debate continues on implementation details.
"""
        elif "writing" in prompt.lower() or "report" in prompt.lower():
            return """
# Research Report

## Executive Summary

This report examines the topic through comprehensive research and analysis.
Key findings indicate significant developments and future opportunities.

## Introduction

### Background
The topic has gained increasing attention in recent years...

### Research Objectives
This study aims to provide comprehensive insights...

## Current State

### Overview
Recent developments show...

### Key Developments
Notable trends include...

## Analysis

### Challenges and Opportunities
The landscape presents both challenges and opportunities...

### Critical Evaluation
Evidence suggests...

## Future Directions

### Emerging Trends
Looking ahead, several trends are emerging...

### Recommendations
Based on the analysis, we recommend...

## Conclusion

### Summary
This research has revealed...

### Final Thoughts
In conclusion...
"""
        elif "quality" in prompt.lower() or "qa" in prompt.lower() or "review" in prompt.lower():
            return """
Quality Assurance Review:

OVERALL ASSESSMENT: Approved with minor suggestions

QUALITY SCORE: 85/100

STRENGTHS:
- Comprehensive coverage of the topic
- Well-structured organization
- Clear and professional writing
- Evidence-based conclusions

ISSUES FOUND:
- Minor: Some sections could benefit from additional detail
- Minor: A few citations could be strengthened

RECOMMENDATIONS:
- Consider expanding the analysis section
- Add more specific examples where applicable
- Ensure all claims are properly supported

FACT-CHECK RESULTS:
- All major claims verified against research data
- No significant contradictions found
- Methodology appears sound

APPROVAL STATUS: APPROVED
"""
        else:
            return f"""
[Mock LLM response from {self.model}]

This is a simulated response for testing purposes.
The system is configured but not making real API calls.

To enable real LLM calls, ensure:
1. Provider SDK is installed (openai or anthropic)
2. Valid API key is configured
3. Network connectivity to the provider

Prompt received: {prompt[:100]}...
"""
