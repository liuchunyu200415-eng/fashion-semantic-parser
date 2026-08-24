"""Serve the local PRD 3.1.2 human image-screening interface."""

import argparse
import secrets
import sys
from pathlib import Path

DEFAULT_PLAN = "data/benchmarks/localization/prd_312_acceptance_screening_v1.json"
DEFAULT_CSV = "outputs/localization/prd_312_acceptance_screening_v1.csv"
DEFAULT_REVIEWED_OUTPUT = (
    "data/benchmarks/localization/prd_312_acceptance_screening_reviewed_v1.json"
)
DEFAULT_SUMMARY = (
    "outputs/localization/prd_312_acceptance_screening_reviewed_summary.json"
)


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse screening artifacts and HTTP binding options.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Serve the model-independent PRD 3.1.2 screening UI."
    )
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--reviewed-output", default=DEFAULT_REVIEWED_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--token")
    return parser.parse_args()


def main() -> None:
    """Load the CSV checkpoint and run the dedicated screening server.

    Raises:
        ValueError: If the port or explicit token is invalid.
    """
    args = parse_args()
    if args.port < 1 or args.port > 65535:
        raise ValueError("Screening port must be in [1, 65535].")
    if args.token is not None and len(args.token) < 16:
        raise ValueError("An explicit screening token must contain 16 characters.")
    add_src_to_python_path()

    import uvicorn

    from fashion_semantic_parser.api.prd_312_acceptance_screening_web import (
        Prd312AcceptanceScreeningWorkspace,
        create_prd_312_acceptance_screening_app,
    )
    from fashion_semantic_parser.common.paths import resolve_project_path

    project_root = Path(__file__).resolve().parents[1]
    public_binding = args.host not in {"127.0.0.1", "localhost", "::1"}
    access_token = args.token
    if public_binding and access_token is None:
        access_token = secrets.token_urlsafe(24)
    workspace = Prd312AcceptanceScreeningWorkspace(
        project_root=project_root,
        plan_path=resolve_project_path(args.plan),
        csv_path=resolve_project_path(args.csv),
        reviewed_output_path=resolve_project_path(args.reviewed_output),
        summary_output_path=resolve_project_path(args.summary_output),
    )
    html_path = project_root / "web" / "prd_312_acceptance_screening.html"
    app = create_prd_312_acceptance_screening_app(
        workspace=workspace,
        html_path=html_path,
        access_token=access_token,
    )
    if access_token is not None:
        print(f"screening_token: {access_token}", flush=True)
        print(
            f"screening_url: http://{args.host}:{args.port}/?token={access_token}",
            flush=True,
        )
    else:
        print(f"screening_url: http://{args.host}:{args.port}/", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
