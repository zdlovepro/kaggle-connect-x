# ConnectX 算法设计文档

## 1. 棋盘表示 — Bitboard

### 1.1 编码方案

使用 `np.uint64` 表示整个棋盘。棋盘 6 行 x 7 列 = 42 格，每列使用 7 位（6 个数据位 + 1 个哨兵位），总共 49 位，可放入 64 位整数。

```
列偏移: ROW_STRIDE = ROWS + 1 = 7

BitBoard 位布局 (stride=7):

Row 0:  .  .  .  .  .  .  .
        0  1  2  3  4  5  6
Row 1:  .  .  .  .  .  .  .
        7  8  9 10 11 12 13
...
Row 5:  .  .  .  .  .  .  .
       35 36 37 38 39 40 41

位置索引: pos = row * 7 + col
```

### 1.2 基础操作

```python
# 落子
board |= np.uint64(1) << pos

# 高度维护 - 每列已填充的棋子数
heights[col] += 1

# 合法着法
valid = [c for c in range(7) if heights[c] < 6]
```

---

## 2. 赢棋检测 — 位运算位移自交

对每个方向，将棋盘与自身移位版本做 AND，重复 3 次后结果非零则表示存在 4 连：

```
水平:   b = board & (board >> 1)  循环 3 次, ~COL6_MASK 滤除跨列
垂直:   b = board & (board >> 7)  循环 3 次
主对角: b = board & (board >> 8)  循环 3 次, ~COL6_MASK 滤除跨列
反对角: b = board & (board >> 6)  循环 3 次, ~COL0_MASK 滤除跨列
```

时间复杂度: O(1)，所有方向通过位运算并行检测。

---

## 3. 搜索算法 — NegaScout (PVS)

### 3.1 核心思路

NegaScout (Principal Variation Search) 是 Alpha-Beta 剪枝的改进：

- 对第一个子节点：全窗口搜索 `[-beta, -alpha]`
- 对后续子节点：空窗口快速测试 `[-alpha-1, -alpha]`
- 若空窗口结果落在区间内 (alpha < score < beta)：用完整窗口重搜

```
function negascout(node, depth, alpha, beta, color):
    if depth == 0 or terminal(node):
        return color * evaluate(node)

    for i, child in enumerate(children):
        if i == 0:
            score = -negascout(child, depth-1, -beta, -alpha, -color)
        else:
            score = -negascout(child, depth-1, -alpha-1, -alpha, -color)  // scout
            if alpha < score < beta:                                       // re-search
                score = -negascout(child, depth-1, -beta, -score, -color)
        alpha = max(alpha, score)
        if alpha >= beta:
            break  // beta cutoff
    return alpha
```

### 3.2 迭代加深

从浅到深逐步搜索，浅层的 TT 结果加速深层搜索：

```
自适应起点:
  empties <= 4  → start_depth = 10  (残局搜索深)
  empties <= 10 → start_depth = 8
  empties <= 14 → start_depth = 6
  否则          → start_depth = 4   (开局搜索浅)

for depth in range(start_depth, MAX_DEPTH + 1):
    score = negascout(..., depth, ...)
    if time_up: break
```

### 3.3 时间管理

- 总预算: 1.9s (Kaggle 限制 2s，留 0.1s 缓冲)
- 每层搜索后检查耗时，超时则中断并返回当前最优结果

---

## 4. 置换表 — Zobrist Hashing

### 4.1 哈希函数

```python
ZOBRIST[pos][player]  # 随机 64 位数, player ∈ {0,1}, pos ∈ [0,41]
hash = 0
for each (pos, player):
    hash ^= ZOBRIST[pos][player]
```

增量更新: 每步只需 `hash ^= ZOBRIST[pos][player-1]`，无需重算。

### 4.2 置换表结构

| 属性 | 值 |
|------|-----|
| 大小 | 1,000,000 条目 |
| 存储 | numpy 数组 (连续内存, 快速访问) |
| 索引 | `hash_key % TT_SIZE` |
| 替换策略 | 深度优先 — 仅当新深度 >= 旧深度时替换 |
| 标志类型 | `EXACT` / `LOWER` / `UPPER` |
| 内容 | `(key, value, depth, flag, best_move)` |

### 4.3 TT Probe 逻辑

```
probe(key, depth, alpha, beta):
    if tt_depth >= depth:
        if flag == EXACT:   return tt_value
        if flag == LOWER:   alpha = max(alpha, tt_value)
        if flag == UPPER:   beta  = min(beta,  tt_value)
        if alpha >= beta:   return tt_value  // cutoff
    return NO_RESULT
```

---

## 5. 着法排序

搜索效率高度依赖着法排序质量。排序优先级从高到低：

| 优先级 | 类型 | 加成分数 | 说明 |
|--------|------|----------|------|
| 1 | TT Best Move | +1,000,000 | 置换表中该局面曾产生最佳结果的着法 |
| 2 | Killer Move | +100,000 | 同深度曾触发 beta cutoff 的着法（每深度 2 槽） |
| 3 | Center Preference | 100 - \|col - 3\| | 中心列优先，向两侧递减 |

好的排序使 beta cutoff 更早发生，剪枝率接近理论最优。

---

## 6. 局面评估函数

### 6.1 传统评估 — `evaluate(b_self, b_opp, heights)`

```
总分 = eval_window_count (2/3 连计数 × 权重)
     + eval_odd_even_threats (奇偶威胁 × 权重)
     + eval_position_heatmap (位置热图 × 权重)

终端: ±10,000,000
```

#### 窗口计数

四个方向（水平/垂直/主对角/反对角）统计连子数：

| 窗口类型 | 权重 | 默认值 |
|----------|------|--------|
| 2-in-a-row | W_SCORE | 10.0 |
| 3-in-a-row | W_THREAT | 100.0 |
| 对手 2-in-a-row | -W_SCORE | -10.0 |
| 对手 3-in-a-row | -W_THREAT | -100.0 |

含双方棋子的窗口 = 0（已死窗口，不影响局面）。

#### 奇偶威胁分析

Connect 4 的核心理论：奇数行威胁 (从底部数 1,3,5) 战略价值远高于偶数行 (2,4,6)。因为奇偶行决定了谁能在该列形成双威胁时掌握主动权。

```
odd-row threat  = 2.0 × phase_scale
even-row threat = 0.5 × phase_scale

phase_scale = filled_cells / 42  (棋盘填充率)
```

威胁权重随对局进程线性增长，残局阶段威胁判定力更强。

#### 位置热图

中心列权重大于两侧，鼓励占据中路战略要地：

```
[3, 4, 5, 7, 5, 4, 3]
[4, 6, 8, 10, 8, 6, 4]
[5, 8, 11, 13, 11, 8, 5]
[5, 8, 11, 13, 11, 8, 5]
[4, 6, 8, 10, 8, 6, 4]
[3, 4, 5, 7, 5, 4, 3]
```

总热图贡献: `W_HMAP(0.3) × 热图值`

### 6.2 学习评估 — `_evaluate_learned()`

线性价值函数近似，由 TD-Learning 训练得到：

```
V(s) = tanh(w · f(s)) × 100,000
```

**11 维特征向量**:

```
[0]  bias (1.0)
[1]  self 2-in-a-row      [2]  self 3-in-a-row
[3]  opp  2-in-a-row      [4]  opp  3-in-a-row
[5]  self heatmap score   [6]  opp  heatmap score
[7]  self odd-row threat  [8]  self even-row threat
[9]  opp  odd-row threat  [10] opp  even-row threat
```

tanh 将输出压缩到 [-1, 1]，乘 100,000 与 NegaScout 搜索的分数范围对齐。

---

## 7. MCTS — 蒙特卡洛树搜索

### 7.1 四阶段流程

```
while time_remaining:
    node = select(root)        // UCB1 从根到叶
    if not terminal(node):
        node = expand(node)     // 展开一个子节点
    result = simulate(node)     // 启发式 rollout 到底
    backpropagate(node, result) // 回溯更新统计

return argmax(children.visits)  // 选访问次数最多的着法
```

### 7.2 UCB1 选择公式

```
UCB1 = win_rate + C × √(ln(parent_visits) / node_visits)
```

`C = 1.414` (默认，可由 Optuna 优化调整)

### 7.3 启发式 Rollout

80% 启发式 + 20% 随机混合，快速模拟至终局：

1. 立即获胜 → 直接获胜列
2. 阻止对手获胜 → 封堵列
3. 延长己方链条 → 最长链方向
4. 中心列优先
5. 随机

MCTS 使用 Python list（非 bitboard）表示棋盘，节点用 `__slots__` 优化内存。

---

## 8. TD-Learning 训练流水线

### 8.1 训练流程

```
for episode in 1..N:
    trajectory = self_play(weights, epsilon)    // ε-greedy, 1-ply 贪心
    weights = td_update(weights, trajectory)     // TD(λ) 权重更新
    if episode % eval_interval == 0:
        inject_weights_into_submission(weights)  // monkey-patch 生产代码
        result = evaluate_vs_random(weights)      // 完整 NegaScout 搜索评估
        if win_rate > best_wr:
            save_best(weights)
        save_checkpoint(state)                   // 断点持久化

return best_weights
```

### 8.2 TD(λ) 更新

```
δ_t = target_t - V(s_t)

target_t = -V(s_{t+1})  (非终端, 视角切换)
        = outcome       (终端, ±1 = 胜/负, 0 = 平)

w ← w + α × δ_t × e_t

e_t = f(s_t) + λ × e_{t-1}  (eligibility trace)
```

| 超参 | 默认值 | 说明 |
|------|--------|------|
| α | 0.005 | 初始学习率 |
| α_decay | 0.9999/局 | 学习率衰减 |
| λ | 0.3 | eligibility trace 衰减因子 |
| ε_start | 0.3 | 初始探索率 |
| ε_end | 0.02 | 最终探索率 |
| eval_interval | 5000 | 评估间隔 (局) |

### 8.3 自对弈

- 1-ply ε-greedy 选择（非完整搜索），保证训练速度
- 硬编码「立即获胜」和「拦截对手」逻辑（不依赖学习）
- 每局生成 ~20-42 步轨迹

### 8.4 权重注入

`_inject_weights_into_submission()` 在运行时修改 `submission.py` 的模块全局变量：

```python
sub._W_SCORE = abs(weights[1]) * 170
sub._W_THREAT = abs(weights[2]) * 170
```

### 8.5 断点续训

```bash
# 首次训练
python evaluate/td_learn.py --episodes 100000

# 中断后恢复 (自动加载 checkpoint)
python evaluate/td_learn.py --resume

# 从指定文件恢复
python evaluate/td_learn.py --load path/to/weights.py
```

Checkpoint 文件 (`agents/td_weights.checkpoint.json`) 包含完整训练状态：
`weights`, `best_weights`, `episode`, `best_win_rate`, `history`

---

## 9. 评估框架

### 9.1 Elo 评分引擎

```
E_A = 1 / (1 + 10^((μ_B - μ_A) / 400))
μ_A' = μ_A + K × (1 - E_A)           // 胜
μ_A' = μ_A + K × (0 - E_A)           // 负

K = K_base × (σ / σ_init)            // 动态 K 因子
σ' = max(σ_min, σ × 0.99)            // 不确定性衰减
95% CI: [μ - 1.96σ, μ + 1.96σ]
```

### 9.2 锦标赛运行器

- **双人对战**: 指定局数，先后手各半
- **循环赛**: N 个 Agent 互相配对，自动 Elo 排名
- **性能基准**: avg / median / min / max / P99 / stdev 单步耗时统计

### 9.3 超参优化 (Optuna)

贝叶斯优化搜索最优评估权重，通过直接 monkey-patch `submission.py` 的全局变量 (`_W_SCORE`, `_W_THREAT`, `_W_HMAP`, `_W_ODD_EVEN`) 注入参数，运行对战评估每个 trial。

```bash
python evaluate/optimize.py --trials 200 --games 50 --opponent negamax
```

### 9.4 消融实验

框架就绪，通过切换组件开关量化各模块贡献。待填充实验列表。

---

## 10. 参数速查表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| W_SCORE | 10.0 | 2-in-a-row 窗口权重 |
| W_THREAT | 100.0 | 3-in-a-row 窗口权重 |
| W_HMAP | 0.3 | 位置热图倍率 |
| W_ODD_EVEN | 80.0 | 奇偶威胁倍率 |
| MAX_DEPTH | 20 | 最大搜索深度 |
| TT_SIZE | 1,000,000 | 置换表条目数 |
| KILLER_SLOTS | 2 | 每深度 Killer 槽位数 |
| TD α | 0.005 | TD 学习率 |
| TD λ | 0.3 | eligibility trace 衰减 |
| ε range | 0.3 → 0.02 | 探索率退火 |
| MCTS C | 1.414 | UCB1 探索常数 |
