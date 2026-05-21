"""
ConnectX Agent - Minimax + Alpha-Beta 剪枝（中级方案）

算法:
  Minimax 搜索 + Alpha-Beta 剪枝
  - 搜索深度: 4 层（开局）/ 动态调整
  - 评估函数: 滑动窗口打分（水平/垂直/对角线）
  - 着法排序: 中心优先 → 提高剪枝效率

时间复杂度: O(b^(d/2))，b=分支因子, d=搜索深度
"""

# --- 棋局基础函数 ---

def _get_valid_actions(board, columns):
    """获取所有未满列的索引"""
    return [c for c in range(columns) if board[c] == 0]


def _drop_piece(board, columns, rows, col, mark):
    """模拟在 col 列落子，返回落子行索引（-1 表示已满）"""
    for r in range(rows - 1, -1, -1):
        if board[r * columns + col] == 0:
            return r
    return -1


def _is_terminal(board, columns, rows, inarow, mark, last_col):
    """
    检测上一步落子(last_col 列)后游戏是否结束。
    返回 (is_over: bool, winner: int)
      winner=mark 表示该方获胜, winner=0 表示游戏继续
    """
    if last_col is None:
        return False, 0

    # 找到最后落子的实际行号（该列最底部非零格）
    last_row = -1
    for r in range(rows - 1, -1, -1):
        idx = r * columns + last_col
        if board[idx] == mark:
            last_row = r
            break
    if last_row == -1:
        return False, 0

    # 4 个方向检测
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        r, c = last_row + dr, last_col + dc
        while 0 <= r < rows and 0 <= c < columns:
            if board[r * columns + c] == mark:
                count += 1
                r += dr
                c += dc
            else:
                break
        r, c = last_row - dr, last_col - dc
        while 0 <= r < rows and 0 <= c < columns:
            if board[r * columns + c] == mark:
                count += 1
                r -= dr
                c -= dc
            else:
                break
        if count >= inarow:
            return True, mark
    return False, 0


def _is_board_full(board, columns):
    """检测棋盘是否已满（平局）"""
    return board[0] != 0


# --- 局面评估函数 ---

def _score_window(window, mark, opponent_mark):
    """
    对长度为 inarow 的窗口打分。
    - 有我方棋子（无对方）→ 正分
    - 有对方棋子（无我方）→ 负分
    - 双方都有 → 0（窗口已死）
    """
    my_count = 0
    opp_count = 0
    for cell in window:
        if cell == mark:
            my_count += 1
        elif cell == opponent_mark:
            opp_count += 1

    if my_count > 0 and opp_count > 0:
        return 0
    if my_count > 0:
        # 指数权重: 1→1, 2→10, 3→100, 4→1000
        return 10 ** (my_count - 1)
    if opp_count > 0:
        return -(10 ** (opp_count - 1))
    return 0


def _evaluate_board(board, columns, rows, inarow, mark, opponent_mark):
    """
    全盘扫描评估：对所有长度为 inarow 的窗口求和。
    窗口方向：水平、垂直、正对角线、反对角线。
    """
    score = 0

    # 水平窗口
    for r in range(rows):
        for c in range(columns - inarow + 1):
            window = [board[r * columns + c + i] for i in range(inarow)]
            score += _score_window(window, mark, opponent_mark)

    # 垂直窗口
    for c in range(columns):
        for r in range(rows - inarow + 1):
            window = [board[(r + i) * columns + c] for i in range(inarow)]
            score += _score_window(window, mark, opponent_mark)

    # 正对角线 (↘)
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board[(r + i) * columns + (c + i)] for i in range(inarow)]
            score += _score_window(window, mark, opponent_mark)

    # 反对角线 (↗)
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            window = [board[(r - i) * columns + (c + i)] for i in range(inarow)]
            score += _score_window(window, mark, opponent_mark)

    # 中心列小幅加成（战略价值）
    center = columns // 2
    for r in range(rows):
        cell = board[r * columns + center]
        if cell == mark:
            score += 2
        elif cell == opponent_mark:
            score -= 2

    return score


# --- Minimax + Alpha-Beta ---

def _order_moves(valid_actions, columns):
    """着法排序：中心列优先，提高剪枝效率"""
    center = columns // 2
    return sorted(valid_actions, key=lambda c: abs(c - center))


def _minimax(board, columns, rows, inarow, depth, alpha, beta,
             maximizing, mark, opponent_mark, last_col):
    """
    Minimax 递归搜索 + Alpha-Beta 剪枝。

    Args:
        last_col: 上一步落子的列号，用于终端检测
    Returns:
        (best_score, best_col) 或 (best_score, None) 叶子节点
    """
    valid = _get_valid_actions(board, columns)
    if not valid:
        return 0, None  # 平局

    # 终端检测
    current_mark = mark if maximizing else opponent_mark
    is_over, winner = _is_terminal(board, columns, rows, inarow, current_mark, last_col)
    if is_over:
        return (10000 if winner == mark else -10000), None

    # 深度耗尽 → 静态评估
    if depth == 0:
        return _evaluate_board(board, columns, rows, inarow, mark, opponent_mark), None

    ordered = _order_moves(valid, columns)

    if maximizing:
        best_score = float("-inf")
        best_col = ordered[0]
        for col in ordered:
            row = _drop_piece(board, columns, rows, col, mark)
            idx = row * columns + col
            board[idx] = mark
            score, _ = _minimax(board, columns, rows, inarow, depth - 1,
                                alpha, beta, False, mark, opponent_mark, col)
            board[idx] = 0
            if score > best_score:
                best_score = score
                best_col = col
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_score, best_col
    else:
        best_score = float("inf")
        best_col = ordered[0]
        for col in ordered:
            row = _drop_piece(board, columns, rows, col, opponent_mark)
            idx = row * columns + col
            board[idx] = opponent_mark
            score, _ = _minimax(board, columns, rows, inarow, depth - 1,
                                alpha, beta, True, mark, opponent_mark, col)
            board[idx] = 0
            if score < best_score:
                best_score = score
                best_col = col
            beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_score, best_col


# --- Agent 入口 ---

def agent(observation, configuration):
    """
    ConnectX Agent 入口函数 (必须为文件中最后一个 callable)

    策略：
      1. 直接获胜 → 立即落子
      2. 拦截对手获胜 → 立即防守
      3. Minimax + Alpha-Beta 搜索（深度自适应）
    """
    board = observation.board
    mark = observation.mark
    columns = configuration.columns
    rows = configuration.rows
    inarow = configuration.inarow
    opponent_mark = 2 if mark == 1 else 1

    valid_actions = _get_valid_actions(board, columns)
    if not valid_actions:
        return 0

    # 快速检测：直接获胜
    for col in valid_actions:
        row = _drop_piece(board, columns, rows, col, mark)
        board[row * columns + col] = mark
        is_over, winner = _is_terminal(board, columns, rows, inarow, mark, col)
        board[row * columns + col] = 0
        if is_over:
            return col

    # 快速检测：拦截对手
    for col in valid_actions:
        row = _drop_piece(board, columns, rows, col, opponent_mark)
        board[row * columns + col] = opponent_mark
        is_over, winner = _is_terminal(board, columns, rows, inarow, opponent_mark, col)
        board[row * columns + col] = 0
        if is_over:
            return col

    # 动态深度：棋子多时搜索更深（空位少→分支少）
    filled = sum(1 for cell in board if cell != 0)
    total = columns * rows
    if filled < total * 0.3:
        depth = 4
    elif filled < total * 0.6:
        depth = 5
    else:
        depth = 6

    _, best_col = _minimax(
        board, columns, rows, inarow, depth,
        float("-inf"), float("inf"), True,
        mark, opponent_mark, None
    )

    return best_col if best_col is not None else valid_actions[0]
