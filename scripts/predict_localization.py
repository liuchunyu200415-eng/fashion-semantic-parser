"""Run one PRD 3.1.2 Grounding DINO + SAM-HQ localization request."""

import argparse
import json
import sys
import time
from pathlib import Path


def add_src_to_python_path() -> None:
    """Add the local package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse one image and natural-language localization query."""
    parser = argparse.ArgumentParser(
        description="Localize one text-described fashion region."
    )
    parser.add_argument("--image", required=True, help="Project-relative image path.")
    parser.add_argument("--query", required=True, help="Natural-language region query.")
    parser.add_argument(
        "--config",
        default="configs/localization_grounded_sam_hq.yaml",
        help="Project-relative Grounding DINO + SAM-HQ YAML.",
    )
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Skip automatic primary-person ROI detection.",
    )
    parser.add_argument(
        "--subject-roi",
        nargs=4,
        type=float,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help="Optional manual subject ROI in original-image xyxy coordinates.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional project-relative JSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the models lazily, run one request, and emit stable JSON."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.models.segmentation import SegmentationSubjectROI
    from fashion_semantic_parser.service.region_localization import (
        GroundedSAMHQRegionLocalizationService,
    )

    if args.subject_roi is not None and args.full_image:
        raise SystemExit("--subject-roi and --full-image cannot be used together.")
    subject_roi = (
        SegmentationSubjectROI(
            x_min=args.subject_roi[0],
            y_min=args.subject_roi[1],
            x_max=args.subject_roi[2],
            y_max=args.subject_roi[3],
        )
        if args.subject_roi is not None
        else None
    )
    auto_subject_roi = subject_roi is None and not args.full_image

    print(
        "状态：正在加载模型并执行语言引导区域定位...",
        file=sys.stderr,
        flush=True,
    )
    start_time = time.perf_counter()
    prediction = GroundedSAMHQRegionLocalizationService(args.config).localize(
        args.image,
        args.query,
        subject_roi=subject_roi,
        auto_subject_roi=auto_subject_roi,
    )
    elapsed_seconds = time.perf_counter() - start_time
    payload = prediction.model_dump(mode="json")
    payload["elapsed_seconds_including_model_load"] = elapsed_seconds
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = resolve_project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    print(
        f"状态：定位完成，返回 {len(prediction.regions)} 个区域，"
        f"首次调用总耗时 {elapsed_seconds:.2f}s。",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
