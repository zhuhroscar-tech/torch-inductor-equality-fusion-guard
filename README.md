[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-inductor-equality-fusion-guard

A call-site workaround and diagnostic for a real `torch.compile(..., backend="inductor")` correctness bug: Inductor can fuse a low-precision (e.g. `bfloat16`) division into a following exact equality/argmax-style comparison **without re-materializing the intermediate dtype's rounding boundary** that eager execution enforces — silently finding fewer matches (often zero) than eager, which corrupts anything downstream that divides by the count. Upstream reference: [pytorch/pytorch#195214](https://github.com/pytorch/pytorch/issues/195214) ("Inductor fuses bf16 division into equality without matching eager's rounding boundary, producing wrong tie-counts and downstream Inf"), root-caused by the reporter to `torch._inductor.config.emulate_precision_casts` defaulting to `False`.

```python
import torch

def tie_count_fn(x, const):
    scaled = x / const                                    # bf16 division
    return (scaled == scaled.amax(dim=-1, keepdim=True)).sum(dim=-1)

x = torch.randn(24, 8, dtype=torch.bfloat16)
eager_tc = tie_count_fn(x, 3.0)                            # correctly finds >=1 tie per row
compiled_tc = torch.compile(tie_count_fn, backend="inductor")(x, 3.0)
# compiled_tc silently finds FEWER ties than eager_tc (often 0) --
# no error, no warning. 1.0 / compiled_tc then produces inf.
```

Independently reproduced from scratch on **this repo's own CI** (both `ubuntu-latest` and `macos-latest`, CPU-only, no GPU runner needed) against the currently installed `torch>=2.2` build — the original upstream report was CUDA-only, but this guard's own verification shows the defect also reproduces on plain CPU Inductor. A `float32` control case is included to confirm this is a precision-dependent fusion defect, not a general Inductor correctness issue.

## Install and check

Requires Python 3.9+ and a compatible PyTorch installation (`torch>=2.2` in the optional extra; NumPy is not required by this guard directly but is pulled in alongside torch for consistency with sibling repos).

```bash
git clone https://github.com/zhuhroscar-tech/torch-inductor-equality-fusion-guard.git
cd torch-inductor-equality-fusion-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[torch]"
torch-inductor-equality-fusion-guard
torch-inductor-equality-fusion-guard --json
```

The CLI reruns 3 cases (two `bfloat16` shapes/constants known to trigger the defect, plus one `float32` control) against the currently installed torch build's Inductor backend, comparing eager vs. default-compiled vs. guarded vs. the `emulate_precision_casts=True` alternative. Its JSON includes the installed torch version, per-case tie counts and match verdicts, `any_default_inductor_diverges_on_bf16`, `guard_fully_correct`, and `emulate_precision_casts_fully_correct`.

Exit codes describe the **guard check**, not just native bug detection: `0` means `precision_safe_division_compare()` matched eager on every case, `1` means it did not, and `2` means torch could not be imported.

## Use in Python

```python
from torch_inductor_equality_fusion_guard import precision_safe_division_compare

divide = precision_safe_division_compare(lambda x, const: x / const)

def tie_count_fn(x, const):
    scaled = divide(x, const)   # always runs eagerly, even under torch.compile
    return (scaled == scaled.amax(dim=-1, keepdim=True)).sum(dim=-1)

compiled = torch.compile(tie_count_fn, backend="inductor")
compiled(x, const)  # now matches eager exactly
```

`precision_safe_division_compare(fn)` wraps `fn` with `torch._dynamo.disable(...)`, forcing it to always run eagerly — with its true, non-fused rounding boundary — even when called from inside an outer `torch.compile`'d region. This is the recommended fix because it is scoped to the single affected call site rather than changing Inductor's fusion behavior for the entire compiled graph.

An equivalent global fix also exists and is verified by this repo's own tests: `torch._inductor.config.patch(emulate_precision_casts=True)` (or setting `torch._inductor.config.emulate_precision_casts = True` globally) makes Inductor re-materialize every low-precision cast boundary, at the cost of applying to the whole compiled graph rather than one call site.

## Scope and limitations

- This tool does **not** patch PyTorch itself. You apply `precision_safe_division_compare` explicitly at your own division-then-comparison call sites, or set the global `emulate_precision_casts` config, the same way you would apply any other workaround.
- The guard's `torch._dynamo.disable` wrapping forces the wrapped region to run eagerly (a graph break at that boundary), which has a real (usually small) performance cost relative to letting Inductor fuse it — this tool does not measure or claim a specific overhead number; profile your own workload if this matters.
- Reproduced and verified against this development host's CPU Inductor backend (macOS arm64 build environment; CI additionally verifies on real `ubuntu-latest` and `macos-latest` GitHub Actions runners). CUDA-specific behavior (the original report's environment) is not independently tested here — no CUDA runner is used, since CPU reproduction was independently confirmed sufficient to exercise the same fusion defect.
- Reproduced and verified only against `torch>=2.2` as resolved by pip at CI build time. If a future released torch version fixes pytorch/pytorch#195214 (e.g. by defaulting `emulate_precision_casts` to `True`), `any_default_inductor_diverges_on_bf16` should report `False` on that version, and this guard remains a safe no-op fallback (still forces eager execution at the wrapped call site, still matches eager by construction).
- This guard is specific to the pattern in the upstream issue (a low-precision division whose result feeds an exact equality/argmax-style comparison). It is not a general audit of every Inductor fusion decision.

## Development

```bash
python -m pip install -e ".[dev,torch]"
python -m pytest -v --cov=torch_inductor_equality_fusion_guard
```

See [implementation](src/torch_inductor_equality_fusion_guard/core.py), [tests](tests/test_core.py), and [CI config](.github/workflows/ci.yml). Licensed under [MIT](LICENSE).
