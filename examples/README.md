# 博客示例

对应[《如何快速寻找史莱姆区块？》](https://yaossg.com/site/docs/slime)。先读正推，再读反推，最后看跨位置复用。

| 阅读顺序 | 文件 | 内容 |
| --- | --- | --- |
| 1 | `cpu_forward.cpp`、`common.hpp` | Java 判定、CPU 按种子枚举位置 |
| 2 | `cuda_seed.cuh` | 每个 CUDA 线程负责一个种子 |
| 3 | `cuda_position.cuh` | 按坐标并行，线程内遍历种子 |
| 4 | `cuda_outer_sum.cuh` | 相邻线程共享坐标轴项 |
| 5 | `reverse.cu`、`reverse_support.cuh`、`grouped.hpp` | 低 18 位筛根、公共项、成组筛选、高 13 位区间求解 |
| 6 | `reuse.cu` | 同坐标类复用根，跳过无根类的位置构造 |

`cuda_steps.cu` 统一运行第 2～4 步并核对完整命中列表。
`brute.cu` 是固定坐标枚举种子的吞吐基线。

## 正推

在仓库根目录执行：

```sh
make examples
./examples/bin/cpu_forward --seed 61692151016558 --seeds 2 --x0 51060 --z0 419115 --width 17 --height 19 -n 5 --repeats 1
./examples/bin/cuda_steps --seed 61692151016558 --seeds 2 --x0 51060 --z0 419115 --width 17 --height 19 -n 5 --repeats 1
```

`--modes 2,3,4` 选择 CUDA 版本；`--help` 查看参数。三版使用相同工作量并比较全部结果。
第 2 步的线程数由种子数决定，少量种子时并行度有限。

## 反推

```sh
./examples/bin/reverse 5
./examples/bin/brute --x 51069 --z 419124 -n 5 --count 17179869184
```

`reverse` 的参数是测量轮数（默认 21）。固定锚点 `(51069,419124)`、实心 5×5，
比较低位筛根后枚举剩余 30 位，以及区间求解、输出差、预计算、成组筛选逐步加入的五版。
每版先预热，之后轮换运行，核对全部 24 个种子。搜索计时包含预计算，不含筛根、分配和结果回传。
朴素剩余位枚举明显更慢；快速验证可用 `./examples/bin/reverse 1`。

`brute` 实际只扫描 `--count` 个种子；完整 48 位耗时须按吞吐外推，不能当作实际完整扫描。

## 坐标类复用

```sh
./examples/bin/reuse examples/data/mixed512.txt 5
./examples/bin/reuse examples/data/random4096.txt 5
```

输入每行是 `x z`，第二个参数为测量轮数。模式 1 是逐位置筛根，模式 2 按坐标类复用根，
模式 3 再跳过无根位置。三版均使用相同的成组筛选与区间求解，完整核对根任务和命中列表。
计时包含分类、筛根、任务展开、传输、搜索和结果回传，不含一次性初始化。

`mixed512.txt` 有 512 个类、每类 64 个位置；`random4096.txt` 是 4096 个不同类的散点，
用于展示没有类复用时的额外开销。结果取决于输入，不能把前者的加速比推广到所有位置分布。

## 正确性

正推测试使用独立 Python Java 算术判定，包含负坐标、溢出、部分 tile 和拒绝重抽。
反推测试比较五版完整结果，并独立验证种子；复用测试比较三版完整根列表与结果。
反推候选仍可能漏掉依赖拒绝重抽的解。
