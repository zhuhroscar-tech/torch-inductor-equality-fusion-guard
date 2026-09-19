[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-inductor-equality-fusion-guard

`torch.compile(..., backend="inductor")` 的一个真实正确性缺陷的调用点变通方案与诊断工具：Inductor 可能会把一次低精度（例如 `bfloat16`）除法直接融合进随后的精确相等/argmax 风格比较中，**跳过 eager 执行本应保留的中间数据类型舍入边界**——从而悄悄找到比 eager 更少的匹配数（常见结果是 0），使任何依赖该计数做后续除法的下游代码产生错误结果。上游参考：[pytorch/pytorch#195214](https://github.com/pytorch/pytorch/issues/195214)（"Inductor 融合 bf16 除法进相等比较时未匹配 eager 的舍入边界，产生错误的并列计数和下游 Inf"），报告者将根因定位到 `torch._inductor.config.emulate_precision_casts` 默认值为 `False`。

```python
import torch

def tie_count_fn(x, const):
    scaled = x / const                                    # bf16 除法
    return (scaled == scaled.amax(dim=-1, keepdim=True)).sum(dim=-1)

x = torch.randn(24, 8, dtype=torch.bfloat16)
eager_tc = tie_count_fn(x, 3.0)                            # 正确地为每行找到 >=1 个并列
compiled_tc = torch.compile(tie_count_fn, backend="inductor")(x, 3.0)
# compiled_tc 悄悄找到比 eager_tc 更少的并列数（常常是 0）——
# 没有报错，没有警告。随后 1.0 / compiled_tc 会产生 inf。
```

已在**本仓库自己的 CI**（`ubuntu-latest` 与 `macos-latest`，仅 CPU，无需 GPU runner）上针对当前安装的 `torch>=2.2` 从零独立复现——上游报告的环境仅限 CUDA，但本工具自己的验证表明该缺陷在纯 CPU Inductor 上同样复现。同时包含一个 `float32` 对照用例，确认这是一个与精度相关的融合缺陷，而非普遍性的 Inductor 正确性问题。

## 安装与检查

需要 Python 3.9+ 及兼容的 PyTorch 安装（可选依赖组中的 `torch>=2.2`；NumPy 并非本工具直接需要，但为与同系列仓库保持一致而一并引入）。

```bash
git clone https://github.com/zhuhroscar-tech/torch-inductor-equality-fusion-guard.git
cd torch-inductor-equality-fusion-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[torch]"
torch-inductor-equality-fusion-guard
torch-inductor-equality-fusion-guard --json
```

CLI 会针对当前安装的 torch 构建的 Inductor 后端重新运行 3 个用例（两个已知触发缺陷的 `bfloat16` 形状/常数组合，外加一个 `float32` 对照），比较 eager、默认编译、加了防护、以及 `emulate_precision_casts=True` 替代方案四者的结果。其 JSON 输出包含已安装的 torch 版本、每个用例的并列计数与匹配结论、`any_default_inductor_diverges_on_bf16`、`guard_fully_correct` 以及 `emulate_precision_casts_fully_correct`。

退出码描述的是**防护检查**而非仅仅是原生缺陷检测：`0` 表示 `precision_safe_division_compare()` 在每个用例上都与 eager 一致，`1` 表示至少有一个不一致，`2` 表示无法导入 torch。

## 在 Python 中使用

```python
from torch_inductor_equality_fusion_guard import precision_safe_division_compare

divide = precision_safe_division_compare(lambda x, const: x / const)

def tie_count_fn(x, const):
    scaled = divide(x, const)   # 即使在 torch.compile 内部，也始终以 eager 方式运行
    return (scaled == scaled.amax(dim=-1, keepdim=True)).sum(dim=-1)

compiled = torch.compile(tie_count_fn, backend="inductor")
compiled(x, const)  # 现在与 eager 完全一致
```

`precision_safe_division_compare(fn)` 用 `torch._dynamo.disable(...)` 包装 `fn`，强制它始终以 eager 方式运行——保留其真实的、未被融合的舍入边界——即使是从外层 `torch.compile` 区域内部调用。这是推荐的修复方式，因为它只作用于单个受影响的调用点，而不会改变整个编译图的 Inductor 融合行为。

本仓库的测试同时验证了一个等效的全局修复方案：`torch._inductor.config.patch(emulate_precision_casts=True)`（或全局设置 `torch._inductor.config.emulate_precision_casts = True`）会让 Inductor 在每一个低精度类型转换边界处重新物化，代价是它作用于整个编译图，而不仅仅是一个调用点。

## 范围与局限性

- 本工具**不会**修补 PyTorch 本身。你需要在自己的除法-比较调用点显式应用 `precision_safe_division_compare`，或设置全局的 `emulate_precision_casts` 配置，就像应用任何其他变通方案一样。
- 防护函数的 `torch._dynamo.disable` 包装会强制被包装的区域以 eager 方式运行（在该边界处产生一次图中断），相较于让 Inductor 融合它，会有真实（通常较小）的性能代价——本工具不测量也不声明具体的开销数字；如果这对你很重要，请在自己的工作负载上进行性能分析。
- 已在本开发主机的 CPU Inductor 后端上复现并验证（macOS arm64 构建环境；CI 另外在真实的 `ubuntu-latest` 与 `macos-latest` GitHub Actions runner 上验证）。CUDA 特有的行为（上游报告的原始环境）在此未被独立测试——由于 CPU 复现已被独立确认足以触发同样的融合缺陷，因此未使用 CUDA runner。
- 仅针对 CI 构建时 pip 实际解析到的 `torch>=2.2` 版本复现并验证。如果未来发布的 torch 版本修复了 pytorch/pytorch#195214（例如将 `emulate_precision_casts` 默认改为 `True`），`any_default_inductor_diverges_on_bf16` 在该版本上应报告 `False`，而本防护仍然是安全的空操作回退（仍会在被包装的调用点强制 eager 执行，因构造原理仍与 eager 一致）。
- 本防护专门针对上游 issue 中的模式（一次低精度除法的结果输入到精确相等/argmax 风格的比较中）。它不是对 Inductor 所有融合决策的通用审计。

## 开发

```bash
python -m pip install -e ".[dev,torch]"
python -m pytest -v --cov=torch_inductor_equality_fusion_guard
```

参见[实现代码](src/torch_inductor_equality_fusion_guard/core.py)、[测试](tests/test_core.py) 与 [CI 配置](.github/workflows/ci.yml)。采用 [MIT](LICENSE) 许可证。
