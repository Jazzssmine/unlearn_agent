import argparse
import sys
from pathlib import Path
from typing import List

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from . import run_influence_baseline as baseline
except ImportError:
    import run_influence_baseline as baseline


DEFAULT_OUTPUT_FILTER_PLACEHOLDER = "[Response filtered due to content policy]"


def _strip_flag_with_value(argv: List[str], flag: str) -> List[str]:
    """Remove all occurrences of `flag <value>` from argv."""
    cleaned: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == flag:
            i += 1
            if i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        cleaned.append(token)
        i += 1
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run influence baseline with a post-generation output filter "
            "(Detoxify threshold + fixed placeholder replacement)."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--output_filter_mode",
        choices=["none", "placeholder"],
        default="placeholder",
        help="Enable/disable post-generation classifier filter.",
    )
    parser.add_argument(
        "--output_filter_placeholder",
        default=DEFAULT_OUTPUT_FILTER_PLACEHOLDER,
        help="Replacement text when generated toxicity exceeds threshold.",
    )
    parser.add_argument(
        "--output_filter_threshold",
        type=float,
        default=None,
        help=(
            "Detoxify threshold for output filter. If provided, overrides "
            "--sanitize_threshold passed to baseline."
        ),
    )

    filter_args, passthrough_argv = parser.parse_known_args()

    # Keep the interface unambiguous: this script controls post-generation
    # output filtering and does not accept direct write-gate controls.
    passthrough_argv = _strip_flag_with_value(passthrough_argv, "--write_gate")
    passthrough_argv = _strip_flag_with_value(passthrough_argv, "--output_filter")

    baseline_argv = [sys.argv[0], *passthrough_argv]

    if filter_args.output_filter_mode == "placeholder":
        baseline_argv.extend(["--write_gate", "redact"])
        baseline.WRITE_REDACTED_TEXT = filter_args.output_filter_placeholder
    else:
        baseline_argv.extend(["--write_gate", "none"])

    if filter_args.output_filter_threshold is not None:
        baseline_argv = _strip_flag_with_value(baseline_argv, "--sanitize_threshold")
        baseline_argv.extend(["--sanitize_threshold", str(filter_args.output_filter_threshold)])

    old_argv = sys.argv
    try:
        sys.argv = baseline_argv
        baseline.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
