"""自动优化的评估权重 (可由 evaluate/optimize.py 重新生成)"""

# 窗口权重: [1-in-row, 2-in-row, 3-in-row, 4-in-row]
WINDOW_WEIGHTS = [1.0, 10.0, 100.0, 1000.0]

CENTER_BONUS = 2.0

# 搜索深度: early / mid / late
DEPTH_EARLY = 4
DEPTH_MID = 5
DEPTH_LATE = 6

# MCTS 参数
MCTS_C = 1.414
MCTS_ROLLOUTS = 800
MCTS_TIME_BUDGET_MS = 1900

# 位置热图 (6行×7列, 对称)
POSITION_HEATMAP = [
    [3, 4, 5, 7, 5, 4, 3],
    [4, 6, 8, 10, 8, 6, 4],
    [5, 8, 11, 13, 11, 8, 5],
    [5, 8, 11, 13, 11, 8, 5],
    [4, 6, 8, 10, 8, 6, 4],
    [3, 4, 5, 7, 5, 4, 3],
]
