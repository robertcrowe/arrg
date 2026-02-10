"""Main entry point for ARRG application."""

import sys
import argparse
from pathlib import Path

def main():
    """Main entry point for ARRG."""
    parser = argparse.ArgumentParser(
        description="ARRG - Automated Research Report Generator"
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        choices=["dashboard", "cli", "version"],
        default="dashboard",
        help="Command to run (default: dashboard)"
    )
    
    parser.add_argument(
        "--topic",
        type=str,
        help="Research topic (for CLI mode)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="claude-haiku-4-5",
        help="Model to use (default: claude-haiku-4-5)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for the model provider"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="Tetrate",
        help="Provider endpoint (default: Tetrate)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (for CLI mode)"
    )
    
    args = parser.parse_args()
    
    if args.command == "version":
        from arrg import __version__
        print(f"ARRG version {__version__}")
        return
    
    if args.command == "dashboard":
        # Launch Streamlit dashboard
        import subprocess
        dashboard_path = Path(__file__).parent / "ui" / "dashboard.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
    
    elif args.command == "cli":
        # Run CLI mode
        if not args.topic:
            print("Error: --topic is required for CLI mode")
            sys.exit(1)
        
        if not args.api_key:
            print("Error: --api-key is required for CLI mode")
            sys.exit(1)
        
        from arrg.core import Orchestrator
        
        print(f"Generating report for: {args.topic}")
        print(f"Using model: {args.model}")
        print("=" * 80)
        
        orchestrator = Orchestrator(
            model=args.model,
            api_key=args.api_key,
            provider_endpoint=args.provider,
            stream_callback=lambda text: print(text),
        )
        
        result = orchestrator.generate_report(args.topic)
        
        if result["status"] == "success":
            report = result["report"]
            
            print("\n" + "=" * 80)
            print("REPORT GENERATED SUCCESSFULLY")
            print("=" * 80)
            print(report["full_text"])
            
            if args.output:
                output_path = Path(args.output)
                output_path.write_text(report["full_text"])
                print(f"\nReport saved to: {output_path}")
        else:
            print(f"\nError: {result.get('error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
