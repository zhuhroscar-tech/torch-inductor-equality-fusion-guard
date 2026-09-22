# torch-inductor-equality-fusion-guard

This repository has been consolidated into [`torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards).

Use the umbrella package instead:

```bash
python -m pip install git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git#egg=torch-correctness-guards[torch]
torch-guard run equality-fusion
```

Python API:

```python
from torch_correctness_guards import precision_safe_division_compare
```

The original diagnostic and tests from this repository now live in the umbrella package as `torch_correctness_guards.guards.equality_fusion` with the CLI name `equality-fusion`.

This source repo is archived to keep the account focused on fewer, maintained packages.
