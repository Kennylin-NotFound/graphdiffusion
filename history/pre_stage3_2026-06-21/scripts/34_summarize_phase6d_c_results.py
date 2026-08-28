"""Generate the Phase 6D-C paper-result report from frozen final evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments.final_reporting import generate_phase6d_c_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("artifacts/phase6d-c-final/final_evidence_freeze.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase6d-c-final/report"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("PHASE6D_C_RESULTS.md"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = args.evidence if args.evidence.is_absolute() else root / args.evidence
    output = args.output if args.output.is_absolute() else root / args.output
    markdown = args.markdown if args.markdown.is_absolute() else root / args.markdown
    manifest = generate_phase6d_c_report(
        evidence,
        output_directory=output,
        markdown_path=markdown,
    )
    print(
        f"scope={manifest['scope']} "
        f"report={manifest['report_json']['path']} "
        f"markdown={manifest['markdown']['path']}"
    )


if __name__ == "__main__":
    main()
