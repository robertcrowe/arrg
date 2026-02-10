def main():
    """Launch ARRG application."""
    import sys
    from arrg.__main__ import main as arrg_main
    
    sys.argv = ["arrg", "dashboard"]  # Default to dashboard
    arrg_main()


if __name__ == "__main__":
    main()
