"""
ConnectX Agent - 启发式规则引擎（初级方案）

决策优先级:
  1. 自己即将获胜 → 落子取胜
  2. 对手即将获胜 → 拦截防守
  3. 优先选择中心列
  4. 选择第一个可用列

时间复杂度: O(columns * inarow) ≈ O(40) 每回合
"""


def _get_valid_actions(board, columns, rows):
    """获取所有可用的列（未满的列）"""
    return [c for c in range(columns) if board[c] == 0]


def _drop_piece(board, columns, rows, col, mark):
    """模拟在指定列落子，返回棋子落在的行索引。若列已满返回 -1"""
    for r in range(rows - 1, -1, -1):
        idx = r * columns + col
        if board[idx] == 0:
            return r
    return -1


def _check_win_at(board, columns, rows, inarow, mark, col, row):
    """
    检测在 (row, col) 落 mark 方棋子后是否获胜。
    从落子位置向 4 个方向扩展计数，
    检查是否存在连续 inarow 个 mark 棋子。
    """
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in directions:
        count = 1

        # 正向扩展
        r, c = row + dr, col + dc
        while 0 <= r < rows and 0 <= c < columns:
            if board[r * columns + c] == mark:
                count += 1
                r += dr
                c += dc
            else:
                break

        # 反向扩展
        r, c = row - dr, col - dc
        while 0 <= r < rows and 0 <= c < columns:
            if board[r * columns + c] == mark:
                count += 1
                r -= dr
                c -= dc
            else:
                break

        if count >= inarow:
            return True

    return False


def _find_winning_move(board, columns, rows, inarow, mark, valid_actions):
    """
    在 valid_actions 中寻找能立即获胜的落子列。
    返回获胜列索引，若不存在则返回 None。
    """
    temp_board = list(board)

    for col in valid_actions:
        row = _drop_piece(temp_board, columns, rows, col, mark)
        if row == -1:
            continue

        idx = row * columns + col
        temp_board[idx] = mark

        if _check_win_at(temp_board, columns, rows, inarow, mark, col, row):
            temp_board[idx] = 0
            return col

        temp_board[idx] = 0

    return None


def agent(observation, configuration):
    """
    ConnectX Agent 入口函数 (必须为文件中最后一个可调用对象)

    Args:
        observation: 环境观察对象
            - board: List[int] 棋盘状态（行优先一维数组）
            - mark: int 当前Agent棋子标识 (1 或 2)
        configuration: 游戏规则配置对象
            - columns: int 棋盘列数
            - rows: int 棋盘行数
            - inarow: int 获胜所需连续棋子数

    Returns:
        int: 选择的列索引 [0, columns)
    """
    board = observation.board
    mark = observation.mark
    columns = configuration.columns
    rows = configuration.rows
    inarow = configuration.inarow
    opponent_mark = 2 if mark == 1 else 1

    valid_actions = _get_valid_actions(board, columns, rows)

    if not valid_actions:
        return 0

    # 1. 检查自己能否获胜
    win_action = _find_winning_move(board, columns, rows, inarow, mark, valid_actions)
    if win_action is not None:
        return win_action

    # 2. 检查是否需要拦截对手
    block_action = _find_winning_move(
        board, columns, rows, inarow, opponent_mark, valid_actions
    )
    if block_action is not None:
        return block_action

    # 3. 优先中心列
    center = columns // 2
    center_actions = sorted(valid_actions, key=lambda c: abs(c - center))

    return center_actions[0]
