# 共用数学实现

- `slime/solver_core.hpp`：Java 整数回绕、48 位 Random 状态、精确史莱姆判定与模式验证。
- `slime/upper_interval.hpp`：输出差必要条件和高 13 位区间求交。

教学示例和搜索 CLI 共用这些实现。候选生成使用首次随机输出的普通分支；
最终验证处理 Java 的拒绝重抽，但不能补回此前漏掉的候选。
