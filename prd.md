# PRD: Automated Research Report Generator (ARRG)

## 1. Purpose
The Automated Research Report Generator (ARRG) is a multi-agent orchestration system designed to produce high-quality, structured research reports from a single user-provided topic. The system utilizes the **A2A (Agent-to-Agent) Protocol** to facilitate standardized communication, task handoffs, and data sharing between specialized AI agents.

## 2. Agent Roles & Model Configuration
The system consists of five core agents. Users must be able to specify a unique **Model String** (e.g., `gpt-4o`, `claude-3-5-sonnet`, `gemini-1.5-pro`, default is `claude-haiku-4-5`) for each agent to optimize for cost, speed, or reasoning capabilities.

* **Planning Agent:** Deconstructs the research topic into a logical outline and specific sub-queries.
* **Research Agent:** Executes web searches and data extraction based on the plan.
* **Analysis Agent:** Synthesizes raw data, identifies cross-source patterns, and extracts key insights.
* **Writing Agent:** Drafts the report in Markdown, ensuring professional tone and flow.
* **Quality Assurance (QA) Agent:** Fact-checks claims against research data and ensures coherence.

## 3. Communication Protocol (A2A)
All agent interactions are governed by the **A2A Protocol**.
* **Interoperability:** Agents communicate via structured A2A messages containing headers (sender/receiver IDs), capability negotiation, and payload.
* **Handoffs:** Tasks move sequentially through the agents, triggered by "TaskComplete" or "Proposal" message types.
* **State Management:** Agents reference a shared workspace for large artifacts (data tables, draft sections) to stay within model context limits.

## 4. Dashboard & User Interface
The dashboard serves as the central control plane for the research process.

### 4.1 Configuration & Credentials
* **Provider Endpoint:** A text input field for the API provider (Defaults to `Tetrate`).
* **API Key:** A secure text input field for the provider’s API key (Defaults to empty string).
* **Model Selection Matrix:** A set of input fields allowing users to define the specific model string for each of the five agents.

### 4.2 Research Execution
* **Topic Input:** A text area for the user to define the research scope and constraints.
* **Live Streaming Console:** A real-time terminal window that displays the "streaming thought process" and raw output of the agent currently executing.
* **Progress Tracking:** A visual indicator showing which stage of the A2A workflow is active.

### 4.3 Output & Logging
* **Report Preview:** A rendered Markdown view of the final report.
* **Export Options:** Buttons to "Save as Markdown" and "Export to PDF."
* **System Log:** A downloadable log file (`research_workflow.log`) containing the full history of A2A messages and raw agent outputs for auditability.

## 5. Functional Requirements
* **Consistency:** The Analysis agent must verify that its findings do not contradict the data gathered by the Research agent.
* **Hallucination Mitigation:** The QA Agent must have the authority to "reject" a draft and send it back to the Writing or Research agents if claims cannot be verified.
* **Context Efficiency:** The system must use the A2A shared workspace to pass references rather than re-sending massive text blocks to prevent context window overflow.

## 6. Success Criteria
* **Standardized Communication:** 100% of agent-to-agent interactions follow the A2A Protocol spec.
* **Transparency:** Users can observe the transition between agents via the streaming console without delay.
* **Auditability:** The generated log file accurately reflects the sequence of events from topic input to final QA.
