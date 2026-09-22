# torch-inductor-equality-fusion-guard

本仓库已合并到 [`torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards)。

请改用统一的伞形包：

```bash
python -m pip install git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git#egg=torch-correctness-guards[torch]
torch-guard run equality-fusion
```

Python API：

```python
from torch_correctness_guards import precision_safe_division_compare
```

本仓库原有的诊断逻辑和测试现在位于伞形包的 `torch_correctness_guards.guards.equality_fusion` 模块中，CLI 名称为 `equality-fusion`。

此源仓库将被归档，以保持账号内项目更少、更集中、更易维护。
