# 如何快速寻找史莱姆区块？

[博客正文](https://yaossg.com/site/docs/slime)的配套代码：从 CPU 暴力枚举、CUDA 并行与 Outer Sum，到低位筛根、输出差、成组筛选和高位区间求解。

| 目录 | 内容 |
| --- | --- |
| [examples/](examples/README.md) | 按博客顺序阅读、运行和比较各步算法 |
| [include/](include/README.md) | 共用的 Random 算术与区间求解 |
| [bench/](bench/README.md) | 支持任意图案和多 GPU 的搜索 CLI |
| [results/](results/README.md) | 博客最终性能表对应的历史计时样本 |

## 构建与验证

需要 Linux/WSL、C++17 编译器、Python 3；CUDA 程序另需 NVIDIA 驱动和 CUDA Toolkit。

```sh
make test-cpu           # 仅 CPU，无需安装 CUDA
make                   # 构建示例和搜索 CLI
make test              # 数学与 CLI 检查，需要 nvcc，无需 GPU
make test-gpu          # GPU 结果与独立 Python 判定对照
```

默认 CUDA 架构为 `86`，同时生成 PTX。其他架构如 H100 使用 `make -B CUDA_ARCH=90`。

```sh
./bench/bin/slime-bench --x 51069 --z 419124 --radius 0 --size 5 --max-results 100
./bench/bin/slime-bench --radius 256 --pattern-file bench/patterns/heart-5x5.txt --max-results 1000
```

反推只生成普通分支候选，可能漏掉依赖 Java 拒绝重抽的解；全部输出经过精确验证。
返回的种子是世界种子的低 48 位，坐标单位是区块，图案位置指左上角。

`make clean` 清理编译产物；临时结果放在已忽略的 `artifacts/` 中。

代码采用 [MIT 许可证](LICENSE)。
