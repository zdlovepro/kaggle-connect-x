"""
ConnectX Agent 本地测试脚本

使用方法（需激活 venv 后运行）:
    python test_agent.py

功能:
  - 基础单元测试（动作合法性、获胜机会识别、拦截防守）
  - 与 random/negamax 对手对战验证
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
            board[r * self.columns + 3] = 1  # 第3列填满
        obs = self._make_obs(board, mark=2)
        for _ in range(5):
            action = agent(obs, self.cfg)
            self.assertNotEqual(action, 3)

    def test_win_opportunity(self):
        """测试能识别并抓住水平四连获胜机会"""
        board = [0] * 42
        # 底部行: [1,1,1,0,2,2,2]
        base = 5 * self.columns
        for c in [0, 1, 2]:
            board[base + c] = self.mark  # mark=1
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
        # 底部行: [0,2,2,2,0,1,1]
        for c in [1, 2, 3]:
            board[base + c] = opponent
        for c in [5, 6]:
            board[base + c] = self.mark
        obs = self._make_obs(board, mark=self.mark)
        action = agent(obs, self.cfg)
        # 应该拦截：对方在列0或列4均可形成四连
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
        # 第2列底部3个 mark 棋子
        for r in [5, 4, 3]:
            board[r * self.columns + 2] = self.mark
        obs = self._make_obs(board, mark=self.mark)
        action = agent(obs, self.cfg)
        self.assertEqual(action, 2, f"应选择第2列垂直获胜，实际选择 {action}")

    def test_diagonal_win(self):
        """测试能识别对角线方向获胜"""
        board = [0] * 42
        cols = self.columns
        # 主对角线: (5,0)=1, (4,1)=1, (3,2)=1
        for i in range(3):
            board[(5 - i) * cols + i] = self.mark
        obs = self._make_obs(board, mark=self.mark)
        action = agent(obs, self.cfg)
        self.assertEqual(action, 3,
                         f"应选择第3列完成主对角线四连，实际选择 {action}")

    def test_fallback_on_full_board(self):
        """测试满列情况下正确 fallback"""
        board = [0] * 42
        # 填满除第0列以外的列顶部
        for c in range(1, self.columns):
            for r in range(self.rows):
                board[r * self.columns + c] = 1
        obs = self._make_obs(board, mark=2)
        action = agent(obs, self.cfg)
        # 仅第0列可用
        self.assertEqual(action, 0)


def run_tournament():
    """对战测试：与 random 和 negamax 对手各打 10 局"""
    from kaggle_environments import make

    print("\n" + "=" * 60)
    print("  对战测试")
    print("=" * 60)

    env = make("connectx", debug=False)

    # 对战 random
    wins = 0
    for i in range(10):
        env.run(["submission.py", "random"])
        # 奖励: agent 0 获胜=1, 失败=0
        last_state = env.steps[-1]
        if last_state[0].reward == 1:
            wins += 1
    print(f"  vs random:      {wins}/10 胜")

    # 对战 negamax
    wins = 0
    for i in range(10):
        env.run(["submission.py", "negamax"])
        last_state = env.steps[-1]
        if last_state[0].reward == 1:
            wins += 1
    print(f"  vs negamax:     {wins}/10 胜")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--tournament":
        run_tournament()
    else:
        unittest.main(verbosity=2)
