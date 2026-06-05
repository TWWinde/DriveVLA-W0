# DriveVLA-W0 文档索引

本目录存放对仓库代码结构、模型 I/O、规模与训练算力的技术分析（基于当前已开源部分）。

| 文档 | 说明 |
|------|------|
| [project_analysis_zh.md](./project_analysis_zh.md) | **主文档**：架构、数据流、训练范式、参数量与算力估算 |

## 快速结论

- **核心思想**：在 Emu3 统一 next-token 框架下，用 **世界模型（预测未来视觉 token）** 提供稠密自监督，再叠加 **驾驶动作** 的 Flow Matching / AR / Q-Former 等头。
- **骨干**：Emu3-8B 级 VLM（`hidden_size=4096`, `32` 层, `vocab≈184k`）。
- **主推推理模型**：`Emu3Pi0`（VLM + 32 层 Action Expert，层间共享注意力，Flow Matching 去噪动作）。
- **官方训练参考**：8× NVIDIA L20（40GB）约 **16 小时**（见根目录 `README.md`）。

更新日期：2026-06-04
