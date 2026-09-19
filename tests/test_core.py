"""Regression tests for torch-inductor-equality-fusion-guard.

These prove:
  1. The bug is real and reproducible from scratch on this host's
     installed torch build: torch.compile(..., backend="inductor")
     on a bf16 division-then-equality-comparison pattern
     (`scaled = x / const; tie_count = (scaled ==
     scaled.amax(-1, keepdim=True)).sum(-1)`) finds fewer ties than
     eager execution, and a subsequent division by that tie_count
     produces non-finite values.
  2. The float32 control case shows no such divergence (this is a
     precision-dependent fusion defect, not a general Inductor
     correctness issue).
  3. `precision_safe_division_compare` is an independently-verified
     fix: wrapping the division with it makes the compiled result
     match eager exactly on every case (bug-injection-style check --
     confirmed the guard has something real to catch before trusting
     it fixes it).
  4. The upstream-documented alternative fix,
     `torch._inductor.config.patch(emulate_precision_casts=True)`,
     also matches eager exactly, corroborating the root cause.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch_inductor_equality_fusion_guard.core import (
    _guarded_tie_count_fn,
    _tie_count_fn,
    diagnose,
    precision_safe_division_compare,
)


class TestDefaultInductorDivergesOnBf16:
    def test_default_inductor_diverges_from_eager_on_bf16(self):
        report = diagnose()
        # Not asserted as an eternal truth: if a future torch release
        # fixes pytorch/pytorch#195214 by changing the default fusion
        # behavior, this should flip to False -- update the
        # README/ledger accordingly rather than treating that as a
        # regression in THIS repo.
        assert report["any_default_inductor_diverges_on_bf16"] is True, (
            "expected the default-Inductor bf16 fusion divergence to "
            f"reproduce on torch {report['torch_version']}; if this now "
            "fails, pytorch/pytorch#195214 may be fixed upstream -- "
            "update the README/ledger accordingly"
        )

    def test_bf16_cases_show_nonfinite_downstream_value(self):
        report = diagnose()
        bf16_cases = [c for c in report["cases"] if c["dtype"] == "torch.bfloat16"]
        assert bf16_cases, "expected at least one bf16 case"
        assert any(c["inductor_default_any_nonfinite_downstream"] for c in bf16_cases), (
            "expected at least one bf16 case to produce a non-finite "
            "downstream value (1.0 / tie_count where tie_count == 0)"
        )

    def test_float32_control_case_does_not_diverge(self):
        report = diagnose()
        f32_cases = [c for c in report["cases"] if c["dtype"] == "torch.float32"]
        assert f32_cases, "expected a float32 control case"
        for c in f32_cases:
            assert c["inductor_default_matches_eager"] is True, (
                f"float32 control case {c['description']!r} unexpectedly "
                "diverged between eager and default-compiled Inductor -- "
                "this control case is meant to show no defect"
            )


class TestGuardIsNotACoincidentalNoOp:
    def test_guard_confirmed_against_diverging_default(self):
        report = diagnose()
        bf16_cases = [c for c in report["cases"] if c["dtype"] == "torch.bfloat16"]
        for c in bf16_cases:
            # Establish the guard has something real to catch: the
            # unguarded default path actually diverged for this case.
            assert c["inductor_default_matches_eager"] is False
            # The guard must match eager exactly on the same input.
            assert c["guarded_matches_eager"] is True, c["description"]

    def test_guard_fully_correct_flag_is_true(self):
        report = diagnose()
        assert report["guard_fully_correct"] is True


class TestEmulatePrecisionCastsAlternativeFix:
    def test_emulate_precision_casts_matches_eager_on_every_case(self):
        report = diagnose()
        assert report["emulate_precision_casts_fully_correct"] is True


class TestPrecisionSafeDivisionCompareDirectly:
    def test_wraps_with_dynamo_disable(self):
        def divide(x, const):
            return x / const

        wrapped = precision_safe_division_compare(divide)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        result = wrapped(x, 2.0)
        assert result.dtype == torch.bfloat16
        assert result.tolist() == (x / 2.0).tolist()

    def test_guarded_function_matches_eager_inside_compiled_region(self):
        torch.manual_seed(42)
        x = torch.randn(16, 4, dtype=torch.bfloat16)
        const = 5.0

        def tie_count_fn(x, const):
            scaled = x / const
            return (scaled == scaled.amax(dim=-1, keepdim=True)).sum(dim=-1)

        eager_tc = tie_count_fn(x, const)

        guarded_division = precision_safe_division_compare(lambda x, const: x / const)

        def guarded_tie_count_fn(x, const):
            scaled = guarded_division(x, const)
            return (scaled == scaled.amax(dim=-1, keepdim=True)).sum(dim=-1)

        torch._dynamo.reset()
        compiled = torch.compile(guarded_tie_count_fn, backend="inductor")
        guarded_tc = compiled(x, const)

        assert guarded_tc.tolist() == eager_tc.tolist()


class TestDiagnose:
    def test_diagnose_default_runs_and_reports_consistent_structure(self):
        report = diagnose()
        assert isinstance(report["torch_version"], str)
        assert report["issue_url"] == "https://github.com/pytorch/pytorch/issues/195214"
        assert len(report["cases"]) == 3

    def test_every_case_has_expected_keys(self):
        report = diagnose()
        expected_keys = {
            "description",
            "dtype",
            "eager_tie_counts",
            "inductor_default_tie_counts",
            "inductor_default_matches_eager",
            "inductor_default_any_nonfinite_downstream",
            "inductor_emulate_precision_tie_counts",
            "inductor_emulate_precision_matches_eager",
            "guarded_tie_counts",
            "guarded_matches_eager",
        }
        for case in report["cases"]:
            assert expected_keys.issubset(case.keys())


class TestGuardedTieCountFnDirectlyInEagerMode:
    """`_guarded_tie_count_fn` was previously only reachable through
    `torch.compile(..., backend="inductor")`, which coverage.py cannot
    see into (Inductor codegens the traced graph rather than
    interpreting the Python body). Calling it directly in eager mode
    proves its own statements execute correctly, independent of
    whether tracing/compilation is involved at all."""

    def test_matches_plain_eager_tie_count(self):
        torch.manual_seed(7)
        x = torch.randn(12, 5, dtype=torch.bfloat16)
        const = 4.0
        guarded_division = precision_safe_division_compare(lambda x, const: x / const)

        result = _guarded_tie_count_fn(guarded_division, x, const)
        expected = _tie_count_fn(x, const)

        assert result.tolist() == expected.tolist()

    def test_returns_tie_counts_not_raw_scaled_values(self):
        # A structural check that the function returns a count per row
        # (reduced along the last dim), not the elementwise comparison
        # itself -- guards against an accidental `.sum(dim=-1)` removal.
        x = torch.tensor([[1.0, 1.0, 2.0], [3.0, 3.0, 3.0]], dtype=torch.bfloat16)
        const = 1.0
        guarded_division = precision_safe_division_compare(lambda x, const: x / const)

        result = _guarded_tie_count_fn(guarded_division, x, const)

        assert result.shape == (2,)
        assert result.tolist() == [1, 3]
