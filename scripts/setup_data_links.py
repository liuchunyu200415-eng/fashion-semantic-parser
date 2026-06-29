"""Create project-relative links to external AutoDL data storage."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


LINKS = {
    "data/raw/fashionai": "data/raw/fashionai",
    "data/raw/deepfashion2": "data/raw/deepfashion2",
    "data/processed/autodl": "data/processed",
    "models/checkpoints/autodl": "models/checkpoints",
    "outputs/autodl": "outputs",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create clean project-relative links to AutoDL data storage."
    )
    parser.add_argument(
        "--data-root",
        default="/root/autodl-tmp",
        help="External data root on the server.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing symlink targets managed by this script.",
    )
    return parser.parse_args()


def create_link(link_path: Path, target_path: Path, replace: bool) -> None:
    """Create a symlink and keep normal Git-tracked directories intact.

    Args:
        link_path: Project-relative symlink path.
        target_path: External target path.
        replace: Whether to replace an existing symlink.

    Raises:
        FileExistsError: If the link path exists and is not replaceable.
    """
    target_path.mkdir(parents=True, exist_ok=True)
    link_path.parent.mkdir(parents=True, exist_ok=True)

    if link_path.is_symlink():
        if replace:
            link_path.unlink()
        else:
            return
    elif link_path.exists():
        raise FileExistsError(
            f"{link_path} already exists and is not a symlink. "
            "Move or remove it manually before linking."
        )

    link_path.symlink_to(target_path, target_is_directory=True)


def main() -> None:
    """Create all server data links."""
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()

    for relative_link, relative_target in LINKS.items():
        create_link(
            PROJECT_ROOT / relative_link,
            data_root / relative_target,
            args.replace,
        )
        print(f"{relative_link} -> {data_root / relative_target}")


if __name__ == "__main__":
    main()
