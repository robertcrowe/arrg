"""Quick integration test to verify critical PRD requirements."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from arrg.core import Orchestrator
from arrg.protocol import TaskStatus


def test_per_agent_models():
    """Test that per-agent model configuration works."""
    print("\n🧪 Testing per-agent model configuration...")
    
    models = {
        "planning": "gpt-4o",
        "research": "claude-3-5-sonnet-20241022",
        "analysis": "gpt-4o-mini",
        "writing": "claude-3-5-haiku-20241022",
        "qa": "gpt-4o",
    }
    
    orchestrator = Orchestrator(
        api_key="test-key",
        provider_endpoint="test-provider",
        models=models,
    )
    
    # Verify each agent has correct model
    assert orchestrator.agents["planning"].model == "gpt-4o", "Planning agent model mismatch"
    assert orchestrator.agents["research"].model == "claude-3-5-sonnet-20241022", "Research agent model mismatch"
    assert orchestrator.agents["analysis"].model == "gpt-4o-mini", "Analysis agent model mismatch"
    assert orchestrator.agents["writing"].model == "claude-3-5-haiku-20241022", "Writing agent model mismatch"
    assert orchestrator.agents["qa"].model == "gpt-4o", "QA agent model mismatch"
    
    print("✅ Per-agent model configuration: PASSED")
    return True


def test_qa_retry_mechanism():
    """Test that QA retry mechanism is configured."""
    print("\n🧪 Testing QA retry mechanism...")
    
    orchestrator = Orchestrator(
        api_key="test-key",
        provider_endpoint="test-provider",
        models={"planning": "gpt-4o"},
    )
    
    # Check default max_qa_retries
    assert hasattr(orchestrator, 'max_qa_retries'), "Missing max_qa_retries attribute"
    assert orchestrator.max_qa_retries >= 1, "Max QA retries should be at least 1"
    assert orchestrator.qa_retry_count == 0, "QA retry count should start at 0"
    
    print("✅ QA retry mechanism: CONFIGURED")
    return True


def test_message_logging():
    """Test that message logging is available."""
    print("\n🧪 Testing message logging...")
    
    orchestrator = Orchestrator(
        api_key="test-key",
        provider_endpoint="test-provider",
        models={"planning": "gpt-4o"},
    )
    
    # Should have get_message_log method
    assert hasattr(orchestrator, 'get_message_log'), "Missing get_message_log method"
    
    # Should return a string log
    log = orchestrator.get_message_log()
    assert isinstance(log, str), "Message log should be a string"
    assert "ARRG WORKFLOW LOG" in log, "Log should have header"
    
    print("✅ Message logging: AVAILABLE")
    return True


def test_writing_agent_revision():
    """Test that writing agent can handle revisions."""
    print("\n🧪 Testing writing agent revision capability...")
    
    from arrg.agents.writing import WritingAgent
    from arrg.protocol import SharedWorkspace
    
    workspace_dir = Path("./test_workspace")
    workspace_dir.mkdir(exist_ok=True)
    workspace = SharedWorkspace(workspace_dir)
    
    agent = WritingAgent(
        agent_id="writing",
        model="gpt-4o",
        workspace=workspace,
        api_key="test-key",
        provider_endpoint="test-provider",
    )
    
    # Check for _revise_report method (A2A Protocol - process_task is the entry point)
    assert hasattr(agent, '_revise_report'), "Writing agent missing _revise_report method"
    assert hasattr(agent, 'process_task'), "Writing agent missing process_task method"
    
    print("✅ Writing agent revision: CAPABLE")
    return True


def test_pdf_export_available():
    """Test that PDF export function exists."""
    print("\n🧪 Testing PDF export availability...")
    
    from arrg.ui.dashboard import generate_pdf
    
    # Check function exists
    assert callable(generate_pdf), "generate_pdf function not found"
    
    # Test with minimal report
    test_report = {
        "title": "Test Report",
        "executive_summary": "Test summary",
        "sections": [{"title": "Section 1", "content": "Content"}],
        "conclusion": "Test conclusion",
        "full_text": "# Test Report\n\nContent here",
        "word_count": 10,
    }
    
    pdf_bytes = generate_pdf(test_report)
    assert isinstance(pdf_bytes, bytes), "PDF should return bytes"
    
    print("✅ PDF export: AVAILABLE")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("ARRG Integration Tests - Critical PRD Requirements")
    print("=" * 60)
    
    tests = [
        test_per_agent_models,
        test_qa_retry_mechanism,
        test_message_logging,
        test_writing_agent_revision,
        test_pdf_export_available,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ {test.__name__}: FAILED - {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All critical PRD requirements verified!")
        return 0
    else:
        print("⚠️ Some tests failed - review implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
