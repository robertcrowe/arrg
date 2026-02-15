"""Check that all critical PRD requirements are implemented."""

import sys

def check_requirements():
    """Check all critical PRD requirements."""
    checks = []
    
    # 1. Per-agent model configuration
    try:
        from arrg.agents.base import BaseAgent
        import inspect
        sig = inspect.signature(BaseAgent.__init__)
        has_model_param = 'model' in sig.parameters
        checks.append(("Per-agent model configuration", has_model_param))
    except Exception as e:
        checks.append(("Per-agent model configuration", False, str(e)))
    
    # 2. Five core agents
    try:
        from arrg.agents import PlanningAgent, ResearchAgent, AnalysisAgent, WritingAgent, QAAgent
        checks.append(("Five core agents exist", True))
    except Exception as e:
        checks.append(("Five core agents exist", False, str(e)))
    
    # 3. A2A Protocol
    try:
        from arrg.a2a import Task, TaskState, Message, Artifact, AgentCard
        from arrg.protocol import SharedWorkspace
        checks.append(("A2A Protocol implemented", True))
    except Exception as e:
        checks.append(("A2A Protocol implemented", False, str(e)))
    
    # 4. Dashboard UI
    try:
        from arrg.ui.dashboard import render_sidebar, render_dashboard
        checks.append(("Dashboard UI exists", True))
    except Exception as e:
        checks.append(("Dashboard UI exists", False, str(e)))
    
    # 5. QA retry mechanism
    try:
        from arrg.core.orchestrator import Orchestrator
        import inspect
        source = inspect.getsource(Orchestrator)
        has_retry = 'max_qa_retries' in source
        checks.append(("QA retry mechanism", has_retry))
    except Exception as e:
        checks.append(("QA retry mechanism", False, str(e)))
    
    # 6. LLM client integration
    try:
        from arrg.utils.llm_client import LLMClient
        checks.append(("LLM client exists", True))
    except Exception as e:
        checks.append(("LLM client exists", False, str(e)))
    
    # 7. Export functionality
    try:
        from arrg.ui.dashboard import export_to_markdown, export_to_pdf
        checks.append(("Export functionality", True))
    except Exception as e:
        checks.append(("Export functionality", False, str(e)))
    
    # 8. Message logging
    try:
        from arrg.agents.base import BaseAgent
        import inspect
        source = inspect.getsource(BaseAgent)
        has_logging = 'message_history' in source
        checks.append(("Message logging", has_logging))
    except Exception as e:
        checks.append(("Message logging", False, str(e)))
    
    # Print results
    print("=" * 60)
    print("PRD Requirements Check")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for check in checks:
        name = check[0]
        status = check[1]
        if status:
            print(f"✅ {name}")
            passed += 1
        else:
            error = check[2] if len(check) > 2 else "Not found"
            print(f"❌ {name}: {error}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed}/{len(checks)} passed")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = check_requirements()
    sys.exit(0 if success else 1)
