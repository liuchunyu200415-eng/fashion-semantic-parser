"""Run a minimal local service smoke check."""

import sys
from pathlib import Path


def add_src_to_python_path() -> None:
    """Add the local src directory when the package is not installed yet."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> None:
    """Load settings and print the configured application name."""
    add_src_to_python_path()

    from fashion_semantic_parser.config import load_settings

    settings = load_settings()
    print(f"Loaded project: {settings.app.name}")


if __name__ == "__main__":
    main()
