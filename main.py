"""Backward-compatible entry point — delegates to ``earsys.cli``."""

from earsys.cli import app

if __name__ == "__main__":
    app()
