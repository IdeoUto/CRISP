# CRISP

**CRISP: An Efficient Cryptographic Framework for ML Inference Against Malicious Clients**

*Xiaoyu Fang, Shihui Zheng, Lize Gu (Beijing University of Posts and Telecommunications)*
NDSS 2026 — https://dx.doi.org/10.14722/ndss.2026.230011

CRISP 是一个抵御恶意客户端的双向安全机器学习推理框架：

- **非线性层**：基于函数秘密共享（FSS）的 ReLU/MaxPool 协议，一次在线交互同时完成非线性函数计算与 MAC 重构，每个非线性层实例的在线通信仅 2n bits；
- **线性层**：利用 CKKS 的复数域编码，把验证标签放在虚部与数据同一次同态运算中完成（复域验证），省去额外的 MAC 同态计算。

## 仓库结构

| 目录 | 内容 |
|---|---|
| `CRISP/` | 非线性层 FSS 实现：DCF 比较函数、认证 ReLU（`bench/`、`test/relu/`、`tests/`） |
| `CKKS-master/` | 线性层 CKKS 复域验证实现与 ring-32 SIMC GC 参考基线（`validated/`） |
| `crisp-retest/` | 实验脚本、训练模型、原始结果与报告（`REPORT.md` 为最终实验结果汇总） |

## 环境要求

- macOS（Apple Silicon，本实验环境）或 Linux；CMake ≥ 3.18
- [Microsoft SEAL 3.6.4](https://github.com/microsoft/SEAL)（安装到 `/usr/local`，或由 CMake 发现）
- OpenSSL（≥ 1.1.1）
- libomp（macOS：`brew install libomp`）
- EMP-tool 工具库（仅 `simc_ring32` GC 参考基线需要，安装方式见 `CKKS-master/README.md`）
- Python 3.10+ 与 `crisp-retest/requirements.txt` 中的依赖（torch、torchvision、numpy、matplotlib 等）

## 构建与测试

```sh
cd crisp-retest
./build.sh
```

构建 CRISP（FSS 组件）与 CKKS-master（线性层组件）全部目标，运行 ctest（`test_fss`、`test_ckks`），并把二进制复制到两个源码目录的 `bin/`。

## 复现实验结果

```sh
cd crisp-retest
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python train_models.py    # 自动下载 MNIST/CIFAR-10（约 400 MB）并训练 A/B 模型
python3 run_suite.py                # ReLU/全连接/卷积/GC 基线/MaxPool 组件基准（每组 3 次）
.venv/bin/python accuracy.py 16     # 全测试集定点模拟 + 16 个样本的真实 CKKS/FSS 数值验证
.venv/bin/python make_report.py     # 从 results/ 重新汇总生成 REPORT.md、图表与 CSV
```

仓库已附带 `crisp-retest/results/`（全部原始结果 JSON/日志）与 `crisp-retest/models/`（训练好的 A/B 模型权重），直接运行 `make_report.py` 即可重新生成报告，无需重跑基准。数据集与 Python 虚拟环境不入库：数据集由 `train_models.py` 自动下载，来源与校验和见 `crisp-retest/DATA_MODEL_PROVENANCE.json`。

## 实验结果

见 [crisp-retest/REPORT.md](crisp-retest/REPORT.md)。

## 引用

```bibtex
@inproceedings{fang2026crisp,
  title={CRISP: An Efficient Cryptographic Framework for ML Inference Against Malicious Clients},
  author={Fang, Xiaoyu and Zheng, Shihui and Gu, Lize},
  booktitle={Network and Distributed System Security (NDSS) Symposium},
  year={2026},
  doi={10.14722/ndss.2026.230011}
}
```

## 许可

本仓库未附加统一许可证；`CKKS-master/` 保留其原有 `LICENSE` 文件。其余代码如需要用于其他用途，请联系作者。
