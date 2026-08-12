"""Tests for the reproducible official SAM-HQ setup entrypoint."""

from pathlib import Path


def test_setup_pins_runtime_and_ignores_only_untracked_caches() -> None:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "setup_sam_hq_proposal_model.sh"
    )
    script = script_path.read_text(encoding="utf-8")

    assert "e696978d60352dc9a26b12631cd91781502c6546" in script
    assert 'timm_version="0.9.16"' in script
    assert script.count("status --porcelain --untracked-files=no") == 2
