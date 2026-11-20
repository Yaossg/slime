# 图案搜索 CLI

搜索曼哈顿半径内每个锚点的有效 48 位种子，支持 NVIDIA 单卡、多卡及自定义图案。
算法推导见[博客](https://yaossg.com/site/docs/slime)，分步骤示例见 [examples/](../examples/README.md)。

## 使用

```sh
make -C bench
./bench/bin/slime-bench --help
./bench/bin/slime-bench --list-devices
./bench/bin/slime-bench --x 51069 --z 419124 --radius 0 --size 5 --max-results 100
./bench/bin/slime-bench --radius 256 --pattern-file bench/patterns/heart-5x5.txt --devices all --max-results 1000
```

`--radius` 必填，包含边界。锚点为图案左上角，图案可以伸出该菱形范围；不裁剪世界边界。
`--size` 支持实心 1～6 方阵，默认 5。小图案可能产生极多匹配。

`--pattern '#?/##'` 或 `--pattern-file FILE` 指定图案：`#` 必须是史莱姆、`.` 必须不是、`?` 忽略。
宽高各 1～64，至少有一个 `#`，不能与 `--size` 同时指定，不旋转或镜像。
正约束参与候选筛选，负约束在输出时验证。以第一个 `#` 的坐标项归一化，返回坐标始终是图案左上角。

[爱心模板](patterns/heart-5x5.txt)的已知种子为 `22428888508636`，锚点 `(-5,-147)`；
单点普通分支共有 1,181 个匹配。[苦力怕模板](patterns/creeper-6x6.txt)是图案输入示例，不保证在指定搜索范围内有解。

## 调度与输出

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--devices` | `0` | 设备索引列表（如 `0,1`）或 `all` |
| `--position-batch` | `32` | 每次调度的锚点数，1～256 |
| `--job-batch` | `2048` | 每次启动的根任务数，1～65536 |
| `--cache-classes` | `4096` | 每设备 FIFO 根缓存，0 禁用 |
| `--repeat` | `1` | 完整扫描轮数，每轮重新筛根 |
| `--max-results` | `0` | 保存的样本上限，始终统计全部匹配 |
| `--start-anchor` | `0` | 按曼哈顿行顺序编号的扫描起点 |
| `--anchor-count` | 剩余全部 | 本次扫描的连续锚点数 |
| `--output` | 无 | 同时保存 JSON，拒绝覆盖已有文件 |
| `--dry-run` | 关闭 | 仅输出范围，不调用 CUDA |

标准输出为 JSON，进度写入标准错误。`matches` 是全部匹配数，`sample_truncated` 标明样本是否截断。
`scan_wall_ms` 包含筛根、传输、搜索和回传，不含一次性设备初始化与显存分配；
`elapsed_ms` 包含初始化、扫描和线程汇总。各设备的阶段时间会重叠，不能直接相加。

普通分支候选生成不包含拒绝重抽补扫。精确验证保证输出正确，但结果不能声明为严格完整目录。

## 代码结构

| 文件 | 职责 |
| --- | --- |
| `slime_bench.cu` | 主入口 |
| `src/model.hpp` | 图案、配置、锚点与坐标项 |
| `src/cli.hpp` | 帮助、参数解析和验证 |
| `src/cuda_support.cuh` | CUDA 检查与设备缓冲区 |
| `src/kernels.cuh` | 筛根、预计算、成组筛选和搜索 |
| `src/worker.cuh` | 单设备批处理和根缓存 |
| `src/runner.cuh` | 多设备调度与结果汇总 |
| `src/report.cuh` | JSON 输出 |

数学实现共用 [include/slime/](../include/README.md)。内部头文件由单个 CUDA 编译单元包含。

```sh
make -C bench test       # 需要 nvcc，无需 GPU
make -C bench test-gpu   # 需要 GPU
```

`testdata/` 只保存回归测试需要的图案、已知种子和预期计数；博客历史计时统一位于 [results/](../results/README.md)。
