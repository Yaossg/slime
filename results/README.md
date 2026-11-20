# 博客测量样本

保留[博客最终性能表](https://yaossg.com/site/docs/slime)对应的历史样本，测于 2026-10-19：
AMD Ryzen 7 5800H、NVIDIA GeForce RTX 3060 Laptop GPU。它们不是本次整理后重新测得的性能。

| 文件 | 内容 | 当前复现入口 |
| --- | --- | --- |
| `reverse.json` | 区间、输出差、预计算和成组筛选的逐轮样本 | `examples/bin/reverse` |
| `reuse.json` | 重复类与不同类输入的三版批次计时样本 | `examples/bin/reuse` |
| `full-search.json` | 正推吞吐样本、单点与半径 512 的 CLI 扫描样本 | `examples/bin/brute`、`bench/bin/slime-bench` |

`reverse.json` 的 `rows` 按博客顺序列出五版算法的历史样本；
其中剩余 30 位枚举为五轮中位数，区间及后三版为另一轮对照的 21 轮中位数。
当前 `reverse` 为便于复现，将五版合并到同一个程序轮换测量，不能据此认为旧数据来自这份整理后的源码。

复用输入保存在 `examples/data/`。历史数据中的计时口径见博客与示例说明；
`samples_ms` 为逐轮时间，`median_ms` 为中位数（均以毫秒计）。
`full-search.json` 的 CLI 样本计完整扫描墙钟，`brute` 的样本计 kernel 时间；
其中 `projected_full48_ms` 是吞吐外推。
不同阶段和不同输入的加速比不可直接相乘。新跑分请另存到 `artifacts/`。

已知图案、种子及预期计数分别放在 `bench/testdata/` 和 `examples/data/known-5x5.json`，
测试不依赖这些历史性能数据。
