# Kaggle ConnectX Agent

**Connect Four 变体智能博弈 Agent** — 多算法实现 + 完整本地评估框架。

---

## 项目结构

```
kaggle-connect-x/
├── submission.py              # Kaggle 提交入口 (Bitboard Minimax + Alpha-Beta + TT)
├── test_agent.py              # 行为测试 + 对战/可视化 CLI (29 项)
├── test_unit.py               # 全面单元测试 (111 项)
├── requirements.txt           # 依赖声明
├── agents/                    # Agent 实现模块
│   ├── base.py                #   BaseAgent 抽象基类
│   ├── minimax_bitboard.py    #   Bitboard Minimax + NegaScout + Zobrist TT
│   ├── mcts_agent.py          #   MCTS (UCB1 + 启发式 Rollout)
│   ├── zobrist.py             #   Zobrist 哈希 + 置换表 (1M 条目)
│   └── weights.py             #   可调参数 (Optuna 自动生成)
├── evaluate/                  # 本地评估框架
│   ├── elo.py                 #   Elo 评分引擎 (动态 K 因子)
│   ├── tournament.py          #   锦标赛运行器 + 性能基准测试
│   ├── ablation.py            #   消融实验模块
│   ├── optimize.py            #   超参优化 (Optuna Bayesian)
│   └── report.py              #   报告生成器 (Text / JSON)
└── .venv/                     # Python 虚拟环境
```

---

## 快速开始

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境 (Windows)
.venv\Scripts\activate

# 激活虚拟环境 (Linux/macOS)
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 运行测试

```bash
# 行为测试 (29 项: Agent 攻防 / MCTS / Bitboard / Elo / 锦标赛)
python test_agent.py

# 全面单元测试 (111 项: 所有模块的辅助函数和类)
python test_unit.py

# 对战测试 (vs random + negamax, 各 10 局)
python test_agent.py --tournament

# 单局逐步可视化
python test_agent.py --visualize random
python test_agent.py --visualize negamax

# 完整评估 (200 局 + Elo 评分 + 报告)
python test_agent.py --full

# 超参优化
python evaluate/optimize.py --trials 200 --games 50 --opponent negamax
```

---

## Agent 架构

### 三层递进策略

```
agent(observation, configuration)
    │
    ├── 1. 快速检测: 立即获胜 → 返回列号
    ├── 2. 快速检测: 拦截对手 → 返回列号
    ├── 3. 开局库:    前 2 步走中心
    └── 4. 迭代加深搜索 → 返回最佳列
            │
            ├── Bitboard 位运算表示 (np.uint64)
            ├── NegaScout (PVS) 搜索
            ├── Alpha-Beta 剪枝 + 着法排序
            ├── Zobrist 哈希 + 置换表 (深度优先替换)
            ├── 局面评估: 窗口评分 + 位置热图 + 威胁计数
            └── 时间感知 + 自适应深度起点
```

### 三种 Agent 实现

| Agent | 文件 | 搜索算法 | 特点 |
|-------|------|----------|------|
| `submission.py` | 根目录 | Minimax + Alpha-Beta | 独立单文件, Kaggle 提交用 |
| `MinimaxBitboardAgent` | `agents/minimax_bitboard.py` | NegaScout + Bitboard | 位运算加速 5-10x, 置换表 |
| `MCTSAgent` | `agents/mcts_agent.py` | MCTS + UCB1 | 启发式 Rollout, 时间预算 |

### 核心算法参数

| 参数 | 值 |
|------|-----|
| 搜索算法 | Minimax + Alpha-Beta (含 NegaScout 变体) |
| 棋盘表示 | Bitboard (np.uint64, 每列 7 bit 编码) |
| 迭代加深 | 自适应起点 4~8 层, 最高 20 层 |
| 置换表 | Zobrist 哈希, 1M 条目, 深度优先替换 |
| 着法排序 | 置换表最佳 → 中心列 → 两侧 |
| 评估函数 | 窗口评分 (1/2/3/4 连) + 位置热图 + 威胁计数 |
| 时间预算 | 1.9s (预留 0.1s 缓冲) |
| 开局库 | 前 2 步走中心 (P1/P2 对称) |

### 局面评估函数

对棋盘四个方向（水平/垂直/对角线）的滑动窗口打分：

| 窗口内己方棋子数 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 得分 | +1 | +10 | +100 | +1000 |

- 含双方棋子的窗口 = 0（已死）
- 位置热图加成：中心列权重大于两侧
- 威胁计数：3 连威胁额外 +50 分

---

## 评估框架

### Elo 评分引擎 (`evaluate/elo.py`)

- 标准 Elo 公式: `E_A = 1 / (1 + 10^((μ_B - μ_A) / 400))`
- 动态 K 因子: `K = K_base * (σ / σ_initial)` — 不确定时调整幅度更大
- 不确定性衰减: `σ = max(σ_min, σ * 0.99)` 每局收敛
- 95% 置信区间: `[μ - 1.96σ, μ + 1.96σ]`
- 历史快照保存

### 锦标赛运行器 (`evaluate/tournament.py`)

- 双人对战 (`run_matchup`): 指定局数, 先后手各半
- 循环赛 (`run_round_robin`): N 个 Agent 互相配对
- 性能基准测试 (`BenchmarkRunner`): avg/median/min/max/P99/stdev 耗时统计

### 消融实验 (`evaluate/ablation.py`)

测试移除单个组件后的表现下降，量化各模块贡献。

### 超参优化 (`evaluate/optimize.py`)

使用 Optuna Bayesian 优化自动搜索最优参数，结果写入 `agents/weights.py`。

---

## 测试结果

### 单元测试: 140/140 通过

| 测试文件 | 测试数 | 覆盖范围 |
|----------|--------|----------|
| `test_agent.py` | 29 | Agent 行为, MCTS, Bitboard, Elo, 锦标赛集成 |
| `test_unit.py` | 111 | 所有辅助函数, Zobrist TT, MCTSNode, BaseAgent, BenchmarkRunner, Reporter, Ablation |

### 对战测试

| 对手 | 胜率 |
|------|------|
| random | 100% (10/10) |
| negamax | 100% (10/10) |

---

## 可视化示例

```
  0   1   2   3   4   5   6
+---+---+---+---+---+---+---+
| . | . | . | . | . | . | . |
+---+---+---+---+---+---+---+
| . | . | . | X | . | . | . |
+---+---+---+---+---+---+---+
| . | . | O | X | O | . | . |
+---+---+---+---+---+---+---+
| . | . | X | O | X | . | . |
+---+---+---+---+---+---+---+
| . | . | O | X | O | . | . |
+---+---+---+---+---+---+---+
| O | . | X | O | X | . | . |
+---+---+---+---+---+---+---+

  X = Agent, O = 对手, . = 空位
  [ 结 果 ]  Agent 获 胜!
```

---

## Kaggle 提交说明

`submission.py` 是独立提交文件, 满足以下约束:

1. **`agent` 函数必须在文件末尾** — Kaggle 通过 `get_last_callable` 识别入口
2. **仅允许单个 `.py` 文件** — 提交时只上传 `submission.py`
3. **响应时限 ≤ 2 秒** — 算法在 1.9s 内自适应停止
4. **内联所有依赖** — 不依赖 `agents/` 或 `evaluate/` 目录

---

## 依赖

```
kaggle-environments>=1.29.0  # 本地环境模拟 (Kaggle 预装)
numpy>=1.24.0                # 位运算加速 (Kaggle 预装)
optuna>=3.0.0                # 超参优化 (仅本地)
```

---

## 参考资料

- [Kaggle ConnectX 竞赛主页](https://www.kaggle.com/competitions/connectx)
- [kaggle-environments GitHub](https://github.com/Kaggle/kaggle-environments)
- [Connect Four 维基百科](https://en.wikipedia.org/wiki/Connect_Four)
- [Alpha-Beta Pruning 算法](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
- [NegaScout 搜索](https://www.chessprogramming.org/NegaScout)
- [Zobrist Hashing](https://www.chessprogramming.org/Zobrist_Hashing)
- [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
