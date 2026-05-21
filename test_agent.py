"""
ConnectX Agent 本地测试脚本

使用方法（需激活 venv 后运行）:
    python test_agent.py                  # 单元测试
    python test_agent.py --tournament     # 对战测试（含结果汇总）
    python test_agent.py --visualize      # 单局对战可视化
"""

import unittest
from unittest.mock import Mock
from submission import agent


class TestAgent(unittest.TestCase):
    """Agent 基础功能测试"""

    def setUp(self):
        self.columns = 7
        self.rows = 6
        self.inarow = 4
        self.mark = 1
        self.cfg = Mock(columns=self.columns, rows=self.rows, inarow=self.inarow)

    def _make_obs(self, board, mark=1):
        return Mock(board=board, mark=mark, remainingOverageTime=120)

    def test_valid_action_range(self):
        """测试返回列索引在合法范围内"""
        obs = self._make_obs([0] * 42)
        for _ in range(10):
            action = agent(obs, self.cfg)
            self.assertIn(action, range(self.columns))

    def test_avoid_full_column(self):
        """测试不选择已满的列"""
        board = [0] * 42
        for r in range(self.rows):
            board[r * self.columns + 3] = 1
        obs = self._make_obs(board, mark=2)
        for _ in range(5):
            action = agent(obs, self.cfg)
            self.assertNotEqual(action, 3)

    def test_win_opportunity(self):
        """测试能识别并抓住水平四连获胜机会"""
        board = [0] * 42
        base = 5 * self.columns
        for c in [0, 1, 2]:
            board[base + c] = self.mark
        for c in [4, 5, 6]:
            board[base + c] = 2
        obs = self._make_obs(board, mark=self.mark)
        action = agent(obs, self.cfg)
        self.assertEqual(action, 3, f"应选择第3列获胜，实际选择 {action}")

    def test_block_opponent(self):
        """测试能识别并拦截对手的获胜威胁"""
        board = [0] * 42
        opponent = 2
        base = 5 * self.columns
        for c in [1, 2, 3]:
            board[base + c] = opponent
        for c in [5, 6]:
            board[base + c] = self.mark
        obs = self._make_obs(board, mark=self.mark)
        action = agent(obs, self.cfg)
        self.assertIn(action, [0, 4], f"应选择0或4拦截对手，实际选择 {action}")

    def test_center_preference(self):
        """测试空棋盘上倾向于选择中心列"""
        obs = self._make_obs([0] * 42)
        action = agent(obs, self.cfg)
        self.assertEqual(action, self.columns // 2,
                         f"空棋盘应选择中心列(3)，实际选择 {action}")

    def test_vertical_win(self):
        """测试能识别垂直方向获胜"""
        board = [0] * 42
        for r in [5, 4, 3]:
            board[r * self.columns + 2] = self.mark
        obs = self._make_obs(board, mark=self.mark)
        action = agent(obs, self.cfg)
        self.assertEqual(action, 2, f"应选择第2列垂直获胜，实际选择 {action}")

    def test_diagonal_win(self):
        """测试能识别对角线方向获胜（考虑重力）"""
        board = [0] * 42
        cols = self.columns
        # 布置三连: (5,0)=1, (4,1)=1, (3,2)=1
        for i in range(3):
            board[(5 - i) * cols + i] = self.mark
        # 用对方棋子填充第3列底部，迫使落子在 row=2
        for r in [3, 4, 5]:
            board[r * cols + 3] = 2
        obs = self._make_obs(board, mark=self.mark)
        action = agent(obs, self.cfg)
        self.assertEqual(action, 3,
                         f"应选择第3列完成对角线四连（(2,3)连接(3,2)(4,1)(5,0)），实际选择 {action}")

    def test_fallback_on_full_board(self):
        """测试满列情况下正确 fallback"""
        board = [0] * 42
        for c in range(1, self.columns):
            for r in range(self.rows):
                board[r * self.columns + c] = 1
        obs = self._make_obs(board, mark=2)
        action = agent(obs, self.cfg)
        self.assertEqual(action, 0)


# ===================== 终端可视化 =====================


def _build_border(cols, part):
    """构建可变列宽边框（ASCII 字符，兼容所有终端）"""
    if part == "top":
        return "+" + "+".join("---" for _ in range(cols)) + "+"
    elif part == "mid":
        return "+" + "+".join("---" for _ in range(cols)) + "+"
    else:
        return "+" + "+".join("---" for _ in range(cols)) + "+"


def render_board(board, columns=7, rows=6, last_col=None, title=None):
    """
    在终端以矩阵形式渲染棋盘。

    X = 玩家1 (Agent),  O = 玩家2 (对手),  . = 空位
    最后落子位置以 [X] 或 [O] 标记。
    """
    P = {0: " . ", 1: " X ", 2: " O "}

    lines = []
    if title:
        lines.append(f"\n  {title}")
        lines.append("")

    # 列标头
    lines.append("  " + "".join(f"{c:^4}" for c in range(columns)))

    # 上边框
    lines.append("  " + _build_border(columns, "top"))

    for r in range(rows):
        cells = []
        for c in range(columns):
            v = board[r * columns + c]
            s = P.get(v, " . ")
            if last_col is not None and c == last_col and v != 0:
                s = "[" + s.strip() + "]"
            cells.append(s)
        lines.append("  |" + "|".join(f"{c:^3}" for c in cells) + "|")
        if r < rows - 1:
            lines.append("  " + _build_border(columns, "mid"))

    lines.append("  " + _build_border(columns, "bot"))
    lines.append("")

    print("\n".join(lines))


def render_result(our_mark, winner):
    """渲染对战结果"""
    print()
    if winner == 0:
        print("  [ 结 果 ]  平 局")
    elif winner == our_mark:
        print("  [ 结 果 ]  Agent 获 胜!")
    else:
        print("  [ 结 果 ]  对 手 获 胜")
    print()


def play_and_visualize(opponent="random", columns=7, rows=6, inarow=4):
    """
    单局对战 + 逐步可视化。

    每一步在终端渲染当前棋盘状态。
    """
    from kaggle_environments import make

    env = make("connectx", configuration={
        "columns": columns, "rows": rows, "inarow": inarow
    }, debug=True)
    env.run(["submission.py", opponent])

    steps = env.steps
    last_col = None

    for step_idx, step in enumerate(steps):
        obs0 = step[0].observation
        obs1 = step[1].observation

        # 确定当前棋盘的 mark 和 board
        if obs0.step > 0:
            board = obs0.board if obs0.board else obs1.board
        else:
            board = [0] * (columns * rows)

        if step_idx == 0:
            render_board(board, columns, rows, title=f"Step {obs0.step:02d} - 初始棋盘")
            continue

        # 确定当前步是谁下的
        if step[0].action is not None:
            last_col = step[0].action
        elif step[1].action is not None:
            last_col = step[1].action

        if board is None or not any(board):
            continue

        render_board(board, columns, rows, last_col=last_col,
                     title=f"Step {obs0.step:02d} - {'Agent(X)' if step[0].action is not None else '对手(O)'} 落子列 {last_col}")

    # 结果
    last = steps[-1]
    our_mark = 1
    if last[0].reward == 1:
        winner = 1
    elif last[1].reward == 1:
        winner = 2
    else:
        winner = 0

    render_result(our_mark, winner)
    return winner


def run_tournament():
    """对战测试：与 random 和 negamax 对手各打 10 局，汇总结果"""
    from kaggle_environments import make

    print("\n" + "=" * 60)
    print("  对 战 测 试")
    print("  Minimax + Alpha-Beta 剪枝 (深度 4~6)")
    print("=" * 60)

    results = {}

    for opponent_name in ["random", "negamax"]:
        wins = 0
        losses = 0
        draws = 0
        last_board = None

        for i in range(10):
            env = make("connectx", debug=False)
            env.run(["submission.py", opponent_name])
            last_step = env.steps[-1]

            reward0 = last_step[0].reward
            if reward0 == 1:
                wins += 1
            elif last_step[1].reward == 1:
                losses += 1
            else:
                draws += 1

            # 保存最后一局的棋盘用于展示
            if i == 9:
                last_board = list(last_step[0].observation.board)

        results[opponent_name] = (wins, losses, draws)
        print(f"\n  vs {opponent_name:<10}  {wins:2d}胜  {losses:2d}负  {draws:2d}平")

    print("\n" + "=" * 60)

    # 汇总
    total_wins = sum(r[0] for r in results.values())
    total_games = sum(sum(r) for r in results.values())
    print(f"\n  总胜率: {total_wins}/{total_games} ({total_wins/total_games*100:.0f}%)")
    print("=" * 60 + "\n")


# ===================== 入口 =====================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--tournament":
        run_tournament()
    elif len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        opp = sys.argv[2] if len(sys.argv) > 2 else "random"
        play_and_visualize(opponent=opp)
    else:
        unittest.main(verbosity=2)
