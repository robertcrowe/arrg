"""Streamlit Dashboard for ARRG - Automated Research Report Generator."""

import streamlit as st
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from arrg.core import Orchestrator
from arrg.a2a import TaskState


# Configure logging
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"arrg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for better troubleshooting
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Log the file location for user reference
logger = logging.getLogger(__name__)
logger.info(f"=== ARRG Session Started ===")
logger.info(f"Log file location: {log_file.absolute()}")
logger.info(f"View logs with: tail -f {log_file.absolute()}")

# Page configuration
st.set_page_config(
    page_title="ARRG - Automated Research Report Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize session state variables."""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "console_output" not in st.session_state:
        st.session_state.console_output = []
    if "report_generated" not in st.session_state:
        st.session_state.report_generated = False
    if "final_report" not in st.session_state:
        st.session_state.final_report = None
    if "qa_results" not in st.session_state:
        st.session_state.qa_results = None
    if "progress" not in st.session_state:
        st.session_state.progress = {
            "planning": TaskState.SUBMITTED.value,
            "research": TaskState.SUBMITTED.value,
            "analysis": TaskState.SUBMITTED.value,
            "writing": TaskState.SUBMITTED.value,
            "qa": TaskState.SUBMITTED.value,
        }


def stream_callback(text: str):
    """Callback function for streaming output from agents."""
    st.session_state.console_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")


def create_orchestrator(models: Dict[str, str], api_key: str, provider: str) -> Orchestrator:
    """Create and return an orchestrator instance."""
    workspace_dir = Path("./workspace")
    workspace_dir.mkdir(exist_ok=True)
    
    return Orchestrator(
        api_key=api_key,
        provider_endpoint=provider,
        models=models,
        workspace_dir=workspace_dir,
        stream_callback=stream_callback,
    )


def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.title("⚙️ Configuration")
    
    # Model Configuration
    st.sidebar.subheader("Model Settings")
    
    provider = st.sidebar.selectbox(
        "Provider",
        options=["Tetrate", "OpenAI", "Anthropic", "Local"],
        index=0,
        help="Select the LLM provider"
    )
    
    # Model selection based on provider
    if provider == "Tetrate":
        model_options = [
            "claude-haiku-4-5",
            "gpt-4o",
            "gpt-4o-mini", 
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "o1",
            "o1-mini",
        ]
    elif provider == "OpenAI":
        model_options = ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini"]
    elif provider == "Anthropic":
        model_options = ["claude-haiku-4-5", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
    else:
        model_options = ["llama-3.2", "mistral", "custom"]
    
    # Per-agent model configuration
    use_per_agent_models = st.sidebar.checkbox(
        "Configure Per-Agent Models",
        value=False,
        help="Assign different models to each agent"
    )
    
    if use_per_agent_models:
        with st.sidebar.expander("🤖 Agent Models", expanded=True):
            models = {
                "planning": st.selectbox("Planning Agent", model_options, key="planning_model"),
                "research": st.selectbox("Research Agent", model_options, key="research_model"),
                "analysis": st.selectbox("Analysis Agent", model_options, key="analysis_model"),
                "writing": st.selectbox("Writing Agent", model_options, key="writing_model"),
                "qa": st.selectbox("QA Agent", model_options, key="qa_model"),
            }
    else:
        default_model = st.sidebar.selectbox(
            "Model (All Agents)",
            options=model_options,
            help="Select the LLM model to use for all agents"
        )
        models = {
            "planning": default_model,
            "research": default_model,
            "analysis": default_model,
            "writing": default_model,
            "qa": default_model,
        }
    
    api_key = st.sidebar.text_input(
        "API Key",
        type="password",
        help="Enter your API key for the selected provider"
    )
    
    st.sidebar.divider()
    
    # Additional Options
    st.sidebar.subheader("Options")
    
    enable_streaming = st.sidebar.checkbox(
        "Enable Live Streaming",
        value=True,
        help="Stream agent output in real-time"
    )
    
    auto_export = st.sidebar.checkbox(
        "Auto-export on Completion",
        value=False,
        help="Automatically export report when generation completes"
    )
    
    st.sidebar.divider()
    
    # About
    with st.sidebar.expander("ℹ️ About ARRG"):
        st.markdown("""
        **Automated Research Report Generator**
        
        A multi-agent system that generates comprehensive research reports using:
        - **Planning Agent**: Creates research plans
        - **Research Agent**: Gathers information
        - **Analysis Agent**: Synthesizes insights
        - **Writing Agent**: Produces polished reports
        - **QA Agent**: Validates quality
        
        All agents communicate via the A2A Protocol.
        """)
    
    return {
        "provider": provider,
        "models": models,
        "api_key": api_key,
        "enable_streaming": enable_streaming,
        "auto_export": auto_export,
    }


def render_progress_tracker(progress: Dict[str, Any]):
    """Render the workflow progress tracker."""
    st.subheader("📊 Workflow Progress")
    
    phases = ["planning", "research", "analysis", "writing", "qa"]
    phase_names = ["Planning", "Research", "Analysis", "Writing", "QA"]
    
    cols = st.columns(len(phases))
    
    for col, phase, name in zip(cols, phases, phase_names):
        with col:
            status = progress.get(phase, TaskState.SUBMITTED.value)
            # Status values are A2A TaskState strings
            if status == TaskState.COMPLETED.value:
                st.success(f"✅ {name}")
            elif status == TaskState.WORKING.value:
                st.info(f"🔄 {name}")
            elif status == TaskState.FAILED.value:
                st.error(f"❌ {name}")
            else:
                st.text(f"⏳ {name}")


def render_console(output: list):
    """Render the live streaming console."""
    st.subheader("📟 Live Console")
    
    console_container = st.container()
    with console_container:
        if output:
            # Show last 50 lines
            console_text = "\n".join(output[-50:])
            st.text_area(
                "Console Output",
                value=console_text,
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )
        else:
            st.info("Console output will appear here when generation starts...")


def render_report_display(report: Dict[str, Any], qa_results: Dict[str, Any]):
    """Render the final report display."""
    st.subheader("📄 Generated Report")
    
    # QA Results Summary
    if qa_results:
        col1, col2, col3 = st.columns(3)
        with col1:
            if qa_results["approved"]:
                st.success(f"✅ Approved")
            else:
                st.warning(f"⚠️ Needs Revision")
        with col2:
            st.metric("Quality Score", f"{qa_results['quality_score']}/100")
        with col3:
            st.metric("Issues Found", qa_results.get("issues_count", 0))
        
        with st.expander("📋 QA Details"):
            st.write("**Strengths:**")
            for strength in qa_results.get("strengths", []):
                st.write(f"- {strength}")
            
            if qa_results.get("issues"):
                st.write("**Issues:**")
                for issue in qa_results["issues"]:
                    st.write(f"- [{issue['severity'].upper()}] {issue['description']}")
            
            st.write("**Recommendations:**")
            for rec in qa_results.get("recommendations", []):
                st.write(f"- {rec}")
    
    st.divider()
    
    # Report Content
    if report:
        # Title
        st.title(report.get("title", "Research Report"))
        
        # Executive Summary
        with st.expander("📝 Executive Summary", expanded=True):
            st.write(report.get("executive_summary", ""))
        
        # Full Report
        st.markdown(report.get("full_text", ""))
        
        # Metadata
        with st.expander("ℹ️ Metadata"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Topic:** {report.get('topic', 'N/A')}")
                st.write(f"**Word Count:** {report.get('word_count', 0)}")
            with col2:
                st.write(f"**Sections:** {len(report.get('sections', []))}")


def render_export_options(report: Dict[str, Any], orchestrator: Optional[Orchestrator] = None):
    """Render export options for the report."""
    st.subheader("💾 Export Report")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        markdown_content = report.get("full_text", "")
        st.download_button(
            label="📥 Markdown",
            data=markdown_content,
            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    
    with col2:
        import json
        json_content = json.dumps(report, indent=2)
        st.download_button(
            label="📥 JSON",
            data=json_content,
            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    
    with col3:
        # PDF Export
        pdf_content = generate_pdf(report)
        st.download_button(
            label="📥 PDF",
            data=pdf_content,
            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    
    with col4:
        # System Log Download
        if orchestrator:
            log_content = orchestrator.get_message_log()
            st.download_button(
                label="📥 System Log",
                data=log_content,
                file_name=f"arrg_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )


def generate_pdf(report: Dict[str, Any]) -> bytes:
    """
    Generate a PDF from the report.
    
    Args:
        report: Report dictionary
        
    Returns:
        PDF content as bytes
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        from io import BytesIO
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='#1f77b4',
            spaceAfter=30,
            alignment=TA_CENTER,
        )
        story.append(Paragraph(report.get("title", "Research Report"), title_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            alignment=TA_JUSTIFY,
            spaceAfter=12,
        )
        story.append(Paragraph(report.get("executive_summary", ""), body_style))
        story.append(Spacer(1, 0.3 * inch))
        
        # Sections
        for section in report.get("sections", []):
            story.append(Paragraph(section["title"], styles['Heading2']))
            story.append(Spacer(1, 0.1 * inch))
            
            # Clean content for PDF
            content = section["content"].replace("\n", "<br/>")
            story.append(Paragraph(content, body_style))
            story.append(Spacer(1, 0.2 * inch))
        
        # Conclusion
        story.append(PageBreak())
        story.append(Paragraph("Conclusion", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(report.get("conclusion", ""), body_style))
        
        # Metadata footer
        story.append(Spacer(1, 0.5 * inch))
        metadata_style = ParagraphStyle(
            'Metadata',
            parent=styles['Normal'],
            fontSize=8,
            textColor='gray',
        )
        metadata_text = f"Generated by ARRG on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Word Count: {report.get('word_count', 0)}"
        story.append(Paragraph(metadata_text, metadata_style))
        
        doc.build(story)
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content
        
    except ImportError:
        # Fallback: Return text content if reportlab not available
        return report.get("full_text", "").encode('utf-8')


def main():
    """Main dashboard application."""
    init_session_state()
    
    # Header
    st.title("📊 ARRG - Automated Research Report Generator")
    st.markdown("*Multi-agent system for generating comprehensive research reports*")
    
    # Show log file location prominently
    log_files = list(Path("./logs").glob("*.log")) if Path("./logs").exists() else []
    if log_files:
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        with st.expander("📋 System Logs", expanded=False):
            st.success(f"**Log file location:** `{latest_log.absolute()}`")
            st.code(f"tail -f {latest_log.absolute()}", language="bash")
            st.markdown("View logs in real-time using the command above, or check the file directly.")
    else:
        st.warning("⚠️ No log files found. Logs will be created in `./logs/` when the application runs.")
    
    # Sidebar configuration
    config = render_sidebar()
    
    # Main content area
    st.divider()
    
    # Topic input section
    st.subheader("🎯 Research Topic")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        topic = st.text_input(
            "Enter your research topic",
            placeholder="e.g., The Impact of Artificial Intelligence on Healthcare",
            label_visibility="collapsed",
        )
    
    with col2:
        generate_button = st.button(
            "🚀 Generate Report",
            use_container_width=True,
            type="primary",
            disabled=not (topic and config["api_key"]),
        )
    
    if not config["api_key"]:
        st.warning("⚠️ Please enter an API key in the sidebar to enable report generation.")
    
    st.divider()
    
    # Progress tracker
    if st.session_state.progress:
        render_progress_tracker(st.session_state.progress)
        st.divider()
    
    # Generate report
    if generate_button and topic and config["api_key"]:
        st.session_state.console_output = []
        st.session_state.report_generated = False
        
        # Create orchestrator with per-agent models
        orchestrator = create_orchestrator(
            models=config["models"],
            api_key=config["api_key"],
            provider=config["provider"],
        )
        
        # Store orchestrator in session state for log access
        st.session_state.orchestrator = orchestrator
        
        # Run generation
        with st.spinner("Generating report..."):
            result = orchestrator.generate_report(topic)
            
            st.session_state.progress = orchestrator.workflow_progress
            
            if result["status"] == "success":
                st.session_state.report_generated = True
                st.session_state.final_report = result["report"]
                st.session_state.qa_results = result["qa_results"]
                st.success("✅ Report generated successfully!")
            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    # Live console
    if config["enable_streaming"] and st.session_state.console_output:
        render_console(st.session_state.console_output)
        st.divider()
    
    # Display report
    if st.session_state.report_generated and st.session_state.final_report:
        render_report_display(
            st.session_state.final_report,
            st.session_state.qa_results,
        )
        st.divider()
        render_export_options(
            st.session_state.final_report,
            orchestrator=st.session_state.orchestrator
        )


if __name__ == "__main__":
    main()
