"""Command-line interface: run the from-scratch diagnosis of the
Inductor bf16-fusion tie-count rounding bug (pytorch/pytorch#195214)
against the currently installed torch build, using the shared
semantic-color design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-inductor-equality-fusion-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "torch.compile(..., backend='inductor') fuses a "
            "low-precision (e.g. bf16) division into a following exact "
            "equality/argmax-style comparison without re-materializing "
            "the intermediate dtype's rounding boundary "
            "(pytorch/pytorch#195214), and verify that "
            "precision_safe_division_compare() and the "
            "emulate_precision_casts=True config both restore an exact "
            "match with eager execution. Never trusts a cached or "
            "previously-reported result, always re-runs the repro on "
            "THIS host's actual installed torch version."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-inductor-equality-fusion-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_default_inductor_diverges_on_bf16"]:
        print(status_headline(style, "fail", "Inductor bf16 fusion divergence reproduced on this host"))
    else:
        print(status_headline(style, "info", "no bf16 fusion divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "precision_safe_division_compare() matches eager on every case"))
    else:
        print(status_headline(style, "fail", "guard did NOT match eager on at least one case"))

    if report["emulate_precision_casts_fully_correct"]:
        print(status_headline(style, "ok", "emulate_precision_casts=True also matches eager on every case"))
    else:
        print(status_headline(style, "warn", "emulate_precision_casts=True did not match eager on at least one case"))

    section("cases (dtype -> eager vs default-inductor vs guarded)")
    for c in report["cases"]:
        default_flag = "MATCH" if c["inductor_default_matches_eager"] else "DIVERGES"
        guard_flag = "guard-ok" if c["guarded_matches_eager"] else "GUARD-FAILED"
        nonfinite_flag = " nonfinite-downstream!" if c["inductor_default_any_nonfinite_downstream"] else ""
        print_fields(
            [
                (
                    c["description"][:48],
                    f"default={default_flag:9s}{nonfinite_flag}  {guard_flag}",
                )
            ]
        )

    return 0 if report["guard_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
