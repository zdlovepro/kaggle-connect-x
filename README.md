# Kaggle ConnectX Agent

基于 **Minimax + Alpha-Beta 剪枝** 的 Connect Four 变体智能博弈 Agent。

---

## 项目结构

```
kaggle-connect-x/
├── .gitignore          # 排除 .venv/、__pycache__/
├── .venv/              # Python 虚拟环境
├── requirements.txt    # 依赖声明
├── submission.py       # Agent 入口（Minimax + Alpha-Beta）
├── test_agent.py       # 测试脚本（单元/对战/可视化）
└── README.md
```

---

## 快速开始

### 环境配置

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
# 单元测试（8 项基础验证）
python test_agent.py

# 对战测试（vs random + negamax，各 10 局）
python test_agent.py --tournament

# 单局逐步可视化
python test_agent.py --visualize random
python test_agent.py --visualize negamax
```

---

## 算法说明

### 架构

```
agent(observation, configuration)
    │
    ├── 1. 快速检测：立即获胜 → 返回列号
    ├── 2. 快速检测：拦截对手 → 返回列号
    └── 3. Minimax + Alpha-Beta 搜索 → 返回最佳列
            │
            ├── 局面评估（滑动窗口打分）
            ├── 着法排序（中心列优先）
            └── Alpha-Beta 剪枝
```

### Minimax 搜索

| 参数 | 值 |
|------|-----|
| 搜索深度 | 动态 4~6 层（随棋盘填充率递增） |
| 分支因子 | ≤ 7（列数） |
| 剪枝 | Alpha-Beta，着法排序 |
| 最坏耗时 | < 500ms（满足 2s 限制） |

### 局面评估函数

对棋盘上所有长度为 `inarow` 的窗口（水平/垂直/对角线）打分：

| 窗口内己方棋子数 | 1 | 2 | 3 | 4 |
|-------------------|---|---|---|-----|
| 得分 | +1 | +10 | +100 | +1000 |

- 含双方棋子的窗口 = 0（已死）
- 空窗口 = 0
- 中心列额外 +2 加成（战略价值）

---

## 测试结果

### 单元测试 (8/8)

| 测试 | 状态 |
|------|------|
| 合法范围校验 | OK |
| 满列避开 | OK |
| 水平获胜识别 | OK |
| 拦截防守 | OK |
| 中心偏好 | OK |
| 垂直获胜识别 | OK |
| 对角线获胜识别 | OK |
| 边界 fallback | OK |

### 对战测试 (vs negamax, 先手)

| 算法版本 | vs random | vs negamax |
|----------|-----------|------------|
| 启发式规则 (v1) | 10/10 胜 | 7/10 胜 |
| Minimax + Alpha-Beta (v2) | 10/10 胜 | **10/10 胜** |

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

  [ 结 果 ]  Agent 获 胜!
```

- `X` = Agent 棋子（玩家1）
- `O` = 对手棋子（玩家2）
- `.` = 空位
- `[X]` / `[O]` = 最后一步落子位置

---

## kaggle_environments 兼容性

`submission.py` 的关键约束：

1. **`agent` 函数必须在文件末尾** — kaggle 通过 `get_last_callable` 识别入口函数
2. **仅允许单个 `.py` 文件** — 提交时只上传 `submission.py`
3. **响应时限 ≤ 2 秒** — 本算法远低于此限制

---

## 依赖

```
kaggle-environments>=1.29.0
numpy>=1.24.0
```

注意：提交到 Kaggle 时无需提供 `requirements.txt`，平台已预装上述库。

---

## 参考资料

- [Kaggle ConnectX 竞赛主页](https://www.kaggle.com/competitions/connectx)
- [kaggle-environments GitHub](https://github.com/Kaggle/kaggle-environments)
- [Connect Four 维基百科](https://en.wikipedia.org/wiki/Connect_Four)
- [Alpha-Beta Pruning 算法](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
