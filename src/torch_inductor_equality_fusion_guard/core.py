"""torch-inductor-equality-fusion-guard core: guard against a real
Inductor correctness bug (pytorch/pytorch#195214) where torch.compile's
Inductor backend fuses a low-precision division into a following exact
equality/argmax-style comparison without re-materializing the
intermediate dtype's rounding boundary that eager execution enforces.

Upstream reference: pytorch/pytorch#195214 ("Inductor fuses bf16
division into equality without matching eager's rounding boundary,
producing wrong tie-counts and downstream Inf"), root-caused by the
reporter to `torch._inductor.config.emulate_precision_casts` being off
by default. Independently reproduced from scratch on THIS host's CPU
backend (no CUDA required -- the original report was CUDA-only, but
this guard's own verification shows the defect also reproduces on
plain CPU Inductor, so it is genuinely testable on ubuntu-latest /
macos-latest CI without a GPU runner).

Pattern: a pattern like

    scaled = x / const          # produces a low-precision (e.g. bf16)
                                  # intermediate
    tie_count = (scaled == scaled.amax(dim=-1, keepdim=True)).sum(-1)

correctly finds every tie under eager execution (the division result
is rounded to bf16 before the comparison, exactly like the reference
value it's compared against), but under `torch.compile(...,
backend="inductor")`, Inductor can fuse the division directly into the
comparison's codegen, comparing at higher intermediate precision than
bf16 actually has -- silently finding FEWER ties than eager (often
zero), which corrupts anything downstream that divides by the count
(a subsequent `1.0 / tie_count` produces `inf`).

This module's guard, `precision_safe_division_compare`, wraps a
division-then-comparison callable with `torch._dynamo.disable(...)`,
forcing it to run eagerly (with its true, non-fused rounding
boundary) even from inside an outer `torch.compile`'d region. This
exactly matches eager's numerics, verified below against every case.
The equivalent global fix,
`torch._inductor.config.patch(emulate_precision_casts=True)`, is also
verified as a working alternative and documented in the README, but
`precision_safe_division_compare` is the recommended fix because it is
scoped to the affected call site instead of changing Inductor's
global fusion behavior for the entire compiled graph.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List


class TorchUnavailableError(RuntimeError):
    """Raised when torch cannot be imported. Kept as a distinct type so
    callers can distinguish "torch isn't installed" from an actual
    diagnostic failure."""


def _import_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise TorchUnavailableError(
            "torch is required for diagnosis and guarding; install the "
            "'torch' extra."
        ) from exc
    return torch


def precision_safe_division_compare(fn: Callable) -> Callable:
    """Wrap a callable containing a low-precision division whose result
    feeds an exact equality/argmax-style comparison, so torch.compile's
    Inductor backend cannot fuse the division into the comparison and
    skip re-materializing the intermediate dtype's rounding boundary
    (pytorch/pytorch#195214). The wrapped callable always runs eagerly
    (via `torch._dynamo.disable`), even when called from inside an
    outer `torch.compile`'d graph, matching eager numerics exactly at
    negligible cost (the wrapped region is typically a small
    elementwise division, not the bulk of the compiled graph)."""
    torch_module = _import_torch()
    return torch_module._dynamo.disable(fn)


@dataclasses.dataclass
class FusionCase:
    description: str
    dtype: str
    eager_tie_counts: List[int]
    inductor_default_tie_counts: List[int]
    inductor_default_matches_eager: bool
    inductor_default_any_nonfinite_downstream: bool
    inductor_emulate_precision_tie_counts: List[int]
    inductor_emulate_precision_matches_eager: bool
    guarded_tie_counts: List[int]
    guarded_matches_eager: bool


def _tie_count_fn(x, const):
    scaled = x / const
    return (scaled == scaled.amax(dim=-1, keepdim=True)).sum(dim=-1)


def _guarded_tie_count_fn(guarded_division: Callable, x, const):
    """The guarded division-then-comparison pattern, extracted to
    module scope (not a closure local to `_run_case`) so it can be
    called directly in eager mode -- bypassing `torch.compile` --  as
    well as compiled. When called only through `torch.compile(...,
    backend="inductor")`, coverage.py cannot see these statements
    execute (Inductor traces and codegens the body rather than
    interpreting it), which previously left this line permanently
    "missed" despite being exercised on every diagnose() call. A
    direct eager-mode call (see
    TestGuardedTieCountFnDirectlyInEagerMode in test_core.py) proves
    the statements execute correctly independent of tracing, the same
    fix pattern applied to torch-addcdiv-stale-scalar-guard's
    `_adam_step_guarded`."""
    scaled = guarded_division(x, const)
    return (scaled == scaled.amax(dim=-1, keepdim=True)).sum(dim=-1)


def _run_case(torch_module, description: str, x, const, expect_divergence: bool) -> FusionCase:
    torch_module._dynamo.reset()
    eager_tc = _tie_count_fn(x, const)

    torch_module._dynamo.reset()
    compiled_default = torch_module.compile(_tie_count_fn, backend="inductor")
    default_tc = compiled_default(x, const)

    downstream_default = 1.0 / default_tc.float()
    any_nonfinite = bool(~torch_module.isfinite(downstream_default).all())

    torch_module._dynamo.reset()
    with torch_module._inductor.config.patch(emulate_precision_casts=True):
        compiled_emulate = torch_module.compile(_tie_count_fn, backend="inductor")
        emulate_tc = compiled_emulate(x, const)

    guarded_division = precision_safe_division_compare(lambda x, const: x / const)

    def guarded_tie_count_fn(x, const):
        return _guarded_tie_count_fn(guarded_division, x, const)

    torch_module._dynamo.reset()
    compiled_guarded = torch_module.compile(guarded_tie_count_fn, backend="inductor")
    guarded_tc = compiled_guarded(x, const)

    eager_list = eager_tc.tolist()
    default_list = default_tc.tolist()
    emulate_list = emulate_tc.tolist()
    guarded_list = guarded_tc.tolist()

    return FusionCase(
        description=description,
        dtype=str(x.dtype),
        eager_tie_counts=eager_list,
        inductor_default_tie_counts=default_list,
        inductor_default_matches_eager=(default_list == eager_list),
        inductor_default_any_nonfinite_downstream=any_nonfinite,
        inductor_emulate_precision_tie_counts=emulate_list,
        inductor_emulate_precision_matches_eager=(emulate_list == eager_list),
        guarded_tie_counts=guarded_list,
        guarded_matches_eager=(guarded_list == eager_list),
    )


def diagnose() -> Dict[str, Any]:
    """Reproduce the Inductor bf16-fusion tie-count bug from scratch
    against the currently installed torch build's CPU Inductor
    backend, and verify both `precision_safe_division_compare` and the
    `emulate_precision_casts=True` global config match eager exactly.
    Never trusts a cached/prior result -- every call re-runs the
    actual repro."""
    torch_module = _import_torch()

    torch_module.manual_seed(0)
    x_bf16_a = torch_module.randn(24, 8, dtype=torch_module.bfloat16)
    torch_module.manual_seed(1)
    x_bf16_b = torch_module.randn(64, 16, dtype=torch_module.bfloat16)
    torch_module.manual_seed(0)
    x_f32 = torch_module.randn(24, 8, dtype=torch_module.float32)

    cases = [
        _run_case(torch_module, "bf16, 24x8, const=3.0", x_bf16_a, 3.0, True),
        _run_case(torch_module, "bf16, 64x16, const=7.0", x_bf16_b, 7.0, True),
        _run_case(torch_module, "float32, 24x8, const=3.0 (control -- no known defect)", x_f32, 3.0, False),
    ]

    any_default_diverges = any(
        (not c.inductor_default_matches_eager) for c in cases if c.dtype == "torch.bfloat16"
    )
    guard_fully_correct = all(c.guarded_matches_eager for c in cases)
    emulate_precision_fully_correct = all(c.inductor_emulate_precision_matches_eager for c in cases)

    return {
        "torch_version": torch_module.__version__,
        "issue_url": "https://github.com/pytorch/pytorch/issues/195214",
        "cases": [dataclasses.asdict(c) for c in cases],
        "any_default_inductor_diverges_on_bf16": any_default_diverges,
        "guard_fully_correct": guard_fully_correct,
        "emulate_precision_casts_fully_correct": emulate_precision_fully_correct,
    }
