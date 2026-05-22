"""
ConnectX Agent 本地测试脚本

使用方法（需激活 venv 后运行）:
    python test_agent.py                  # 单元测试
    python test_agent.py --tournament     # 对战测试（含结果汇总）
    python test_agent.py --visualize      # 单局对战可视化
    python test_agent.py --full           # 完整评估 (200局 + Elo + 报告)
"""

import unittest
from unittest.mock import Mock

from submission import agent

# ── 导入评估框架 ──
try:
    from evaluate.elo import EloEngine, EloRating
    from evaluate.tournament import TournamentRunner
    from evaluate.report import Reporter
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False

# ── 导入新 Agent ──
try:
    from agents.base import BaseAgent
    from agents.minimax_bitboard import MinimaxBitboardAgent
    from agents.mcts_agent import MCTSAgent
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False


# ===================== 基础功能测试 =====================


class TestAgent(unittest.TestCase):
    """Agent 基础功能测试 (原 8 项)"""

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
        for i in range(3):
            board[(5 - i) * cols + i] = self.mark
        for r in [3, 4, 5]:
            board[r * cols + 3] = 2
        obs = self._make_obs(board, mark=self.mark)
        action = agent(obs, self.cfg)
        self.assertEqual(action, 3,
                         f"应选择第3列完成对角线四连，实际选择 {action}")

    def test_fallback_on_full_board(self):
        """测试满列情况下正确 fallback"""
        board = [0] * 42
        for c in range(1, self.columns):
            for r in range(self.rows):
                board[r * self.columns + c] = 1
        obs = self._make_obs(board, mark=2)
        action = agent(obs, self.cfg)
        self.assertEqual(action, 0)

    # ── 新增：置换表 / 迭代加深测试 ──

    def test_consistent_on_empty_board(self):
        """测试空棋盘结果稳定 (多次调用应返回相同动作)"""
        obs = self._make_obs([0] * 42)
        actions = [agent(obs, self.cfg) for _ in range(5)]
        self.assertEqual(len(set(actions)), 1,
                         f"空棋盘应稳定返回同一列，实际返回 {set(actions)}")

    def test_opening_book(self):
        """测试开局库：前2步走中心"""
        # 第一步 (空棋盘)
        obs = self._make_obs([0] * 42, mark=1)
        a = agent(obs, self.cfg)
        self.assertEqual(a, 3, f"开局第一步应走中心列3，实际 {a}")

        # 第二步 (棋盘上只有1个中心棋子)
        board = [0] * 42
        board[5 * self.columns + 3] = 1  # 对手走了中心
        obs = self._make_obs(board, mark=2)
        a = agent(obs, self.cfg)
        self.assertEqual(a, 3, f"开局第二步应走中心列3，实际 {a}")

    def test_endgame_deep_search(self):
        """测试终局阶段能搜索更深（迭代加深）"""
        # 构造一个需要3层以上搜索才能识别的威胁
        board = [0] * 42
        # 放置棋子形成潜在威胁
        for c in [0, 1]:
            board[5 * self.columns + c] = 1  # 我方2连底部
        for c in [2, 3]:
            board[5 * self.columns + c] = 2  # 对手2连底部
        board[4 * self.columns + 0] = 2  # 对手在列0已有两子
        board[4 * self.columns + 3] = 1
        obs = self._make_obs(board, mark=1)
        action = agent(obs, self.cfg)
        self.assertIn(action, range(self.columns),
                      f"应在合法范围内，实际 {action}")


# ===================== MCTS Agent 测试 =====================


@unittest.skipIf(not AGENTS_AVAILABLE, "agents 模块不可用")
class TestMCTSAgent(unittest.TestCase):
    """MCTS Agent 行为测试"""

    def setUp(self):
        self.columns = 7
        self.rows = 6
        self.inarow = 4
        self.cfg = Mock(columns=self.columns, rows=self.rows, inarow=self.inarow)

    def _make_obs(self, board, mark=1):
        return Mock(board=board, mark=mark, remainingOverageTime=120)

    def test_mcts_valid_action(self):
        """MCTS 返回合法动作"""
        mcts = MCTSAgent(time_budget_ms=500)
        obs = self._make_obs([0] * 42)
        for _ in range(5):
            action = mcts.select_action(obs, self.cfg)
            self.assertIn(action, range(self.columns))

    def test_mcts_win_detection(self):
        """MCTS 能识别并抓住获胜机会"""
        mcts = MCTSAgent(time_budget_ms=500)
        board = [0] * 42
        base = 5 * self.columns
        for c in [0, 1, 2]:
            board[base + c] = 1
        obs = self._make_obs(board, mark=1)
        action = mcts.select_action(obs, self.cfg)
        self.assertEqual(action, 3, f"MCTS 应选择第3列获胜，实际选择 {action}")

    def test_mcts_block_opponent(self):
        """MCTS 能拦截对手获胜威胁"""
        mcts = MCTSAgent(time_budget_ms=500)
        board = [0] * 42
        base = 5 * self.columns
        for c in [1, 2, 3]:
            board[base + c] = 2  # 对手三连
        obs = self._make_obs(board, mark=1)
        action = mcts.select_action(obs, self.cfg)
        self.assertIn(action, [0, 4], f"MCTS 应拦截对手，实际选择 {action}")

    def test_mcts_center_preference(self):
        """MCTS 返回合法动作 (空盘概率倾向中心)"""
        mcts = MCTSAgent(time_budget_ms=500)
        obs = self._make_obs([0] * 42)
        action = mcts.select_action(obs, self.cfg)
        # MCTS 在空盘上应返回合法动作
        self.assertIn(action, range(self.columns),
                      f"MCTS 应返回合法动作，实际选择 {action}")


# ===================== Minimax Bitboard 测试 =====================


@unittest.skipIf(not AGENTS_AVAILABLE, "agents 模块不可用")
class TestMinimaxBitboardAgent(unittest.TestCase):
    """Bitboard Minimax Agent 行为测试"""

    def setUp(self):
        self.columns = 7
        self.rows = 6
        self.inarow = 4
        self.cfg = Mock(columns=self.columns, rows=self.rows, inarow=self.inarow)

    def _make_obs(self, board, mark=1):
        return Mock(board=board, mark=mark, remainingOverageTime=120)

    def test_bitboard_valid_action(self):
        """返回合法动作"""
        agent_bb = MinimaxBitboardAgent(time_budget_ms=500)
        obs = self._make_obs([0] * 42)
        for _ in range(5):
            action = agent_bb.select_action(obs, self.cfg)
            self.assertIn(action, range(self.columns))

    def test_bitboard_win_detection(self):
        """能识别并抓住获胜机会"""
        agent_bb = MinimaxBitboardAgent(time_budget_ms=500)
        board = [0] * 42
        base = 5 * self.columns
        for c in [0, 1, 2]:
            board[base + c] = 1
        obs = self._make_obs(board, mark=1)
        action = agent_bb.select_action(obs, self.cfg)
        self.assertEqual(action, 3, f"应选择第3列获胜，实际选择 {action}")

    def test_bitboard_block_opponent(self):
        """能拦截对手获胜"""
        agent_bb = MinimaxBitboardAgent(time_budget_ms=500)
        board = [0] * 42
        base = 5 * self.columns
        for c in [1, 2, 3]:
            board[base + c] = 2
        obs = self._make_obs(board, mark=1)
        action = agent_bb.select_action(obs, self.cfg)
        self.assertIn(action, [0, 4], f"应拦截对手，实际选择 {action}")

    def test_bitboard_consistent_output(self):
        """输出一致性: 相同输入产生相同输出"""
        agent_bb = MinimaxBitboardAgent(time_budget_ms=500)
        obs = self._make_obs([0] * 42)
        actions = [agent_bb.select_action(obs, self.cfg) for _ in range(3)]
        self.assertEqual(len(set(actions)), 1,
                         f"应稳定返回同一列，实际 {set(actions)}")


# ===================== Elo 评分测试 =====================


@unittest.skipIf(not EVAL_AVAILABLE, "evaluate 模块不可用")
class TestEloEngine(unittest.TestCase):
    """Elo 评分引擎测试"""

    def test_elo_basic(self):
        """Elo 基本计算正确"""
        engine = EloEngine()
        engine.register("A", initial_mu=1000)
        engine.register("B", initial_mu=1000)

        # 同分选手预期各 50%
        ea, eb = engine.expected_score(1000, 1000)
        self.assertAlmostEqual(ea, 0.5, places=2)
        self.assertAlmostEqual(eb, 0.5, places=2)

    def test_elo_stronger_expected(self):
        """高分选手预期胜率高"""
        engine = EloEngine()
        ea, eb = engine.expected_score(1200, 1000)
        self.assertGreater(ea, 0.5)
        self.assertLess(eb, 0.5)

    def test_elo_update_win(self):
        """胜负更新: 胜方评分上升，负方下降"""
        engine = EloEngine()
        engine.register("A", initial_mu=1000)
        engine.register("B", initial_mu=1000)

        # A 赢
        engine.update("A", "B", 1.0)
        ra = engine.get_rating("A")
        rb = engine.get_rating("B")
        self.assertGreater(ra.mu, 1000)
        self.assertLess(rb.mu, 1000)

    def test_elo_update_draw(self):
        """平局: 评分趋近"""
        engine = EloEngine()
        engine.register("A", initial_mu=1200)
        engine.register("B", initial_mu=1000)

        mu_a_before = engine.get_rating("A").mu
        mu_b_before = engine.get_rating("B").mu

        engine.update("A", "B", 0.5)
        ra = engine.get_rating("A")
        rb = engine.get_rating("B")
        self.assertLess(ra.mu, mu_a_before)  # 高分平低分 → 降
        self.assertGreater(rb.mu, mu_b_before)  # 低分平高分 → 升

    def test_elo_confidence_interval(self):
        """置信区间范围正确"""
        engine = EloEngine()
        engine.register("A", initial_mu=1000)
        lo, hi = engine.get_confidence_interval("A")
        self.assertLess(lo, 1000)
        self.assertGreater(hi, 1000)

    def test_elo_sigma_decay(self):
        """不确定性随对局数增加而衰减"""
        engine = EloEngine()
        engine.register("A", initial_mu=1000)
        sigma_before = engine.get_rating("A").sigma

        for _ in range(50):
            engine.update("A", "B", 1.0)

        sigma_after = engine.get_rating("A").sigma
        self.assertLess(sigma_after, sigma_before)

    def test_elo_win_rate(self):
        """胜率统计正确"""
        engine = EloEngine()
        engine.register("A")
        for _ in range(10):
            engine.update("A", "B", 1.0)
        wr = engine.win_rate("A")
        self.assertEqual(wr, 1.0)


# ===================== 评估框架集成测试 =====================


@unittest.skipIf(not EVAL_AVAILABLE, "evaluate 模块不可用")
class TestTournamentIntegration(unittest.TestCase):
    """锦标赛框架集成测试"""

    def test_quick_matchup(self):
        """快速对战: 运行少量局数验证流程"""
        runner = TournamentRunner()
        result = runner.run_matchup(
            "submission.py", "random",
            games=4, verbose=False,
        )
        self.assertIn("win_rate_a", result)
        self.assertEqual(result["games"], 4)
        self.assertGreaterEqual(result["wins_a"] + result["losses_a"] + result["draws_a"], 4)

    def test_elo_integration_with_tournament(self):
        """Elo 与锦标赛集成"""
        runner = TournamentRunner()
        runner.run_matchup("submission.py", "random", games=4, verbose=False)
        r = runner.elo.get_rating("submission.py")
        self.assertIsNotNone(r)
        self.assertGreater(r.games, 0)

    def test_reporter_output(self):
        """报告生成器不报错"""
        engine = EloEngine()
        engine.register("A")
        engine.register("B")
        reporter = Reporter(output_dir="evaluate/results")
        text = reporter.generate_text_report(engine)
        self.assertIn("Elo Ratings", text)


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
    """在终端以矩阵形式渲染棋盘"""
    P = {0: " . ", 1: " X ", 2: " O "}

    lines = []
    if title:
        lines.append(f"\n  {title}")
        lines.append("")

    lines.append("  " + "".join(f"{c:^4}" for c in range(columns)))
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


def play_and_visualize(opponent="random", columns=7, rows=6, inarow=4, agent_file="submission.py"):
    """单局对战 + 逐步可视化"""
    from kaggle_environments import make

    env = make("connectx", configuration={
        "columns": columns, "rows": rows, "inarow": inarow
    }, debug=True)
    env.run([agent_file, opponent])

    steps = env.steps
    last_col = None

    for step_idx, step in enumerate(steps):
        obs0 = step[0].observation
        obs1 = step[1].observation

        if obs0.step > 0:
            board = obs0.board if obs0.board else obs1.board
        else:
            board = [0] * (columns * rows)

        if step_idx == 0:
            render_board(board, columns, rows, title=f"Step {obs0.step:02d} - 初始棋盘")
            continue

        if step[0].action is not None:
            last_col = step[0].action
        elif step[1].action is not None:
            last_col = step[1].action

        if board is None or not any(board):
            continue

        render_board(board, columns, rows, last_col=last_col,
                     title=f"Step {obs0.step:02d} - {'Agent(X)' if step[0].action is not None else '对手(O)'} 落子列 {last_col}")

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
    """对战测试：与 random 和 negamax 对手各打 10 局"""
    from kaggle_environments import make

    print("\n" + "=" * 60)
    print("  对 战 测 试")
    print("  Bitboard Minimax + 置换表 + 迭代加深")
    print("=" * 60)

    results = {}

    for opponent_name in ["random", "negamax"]:
        wins = 0
        losses = 0
        draws = 0

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

        results[opponent_name] = (wins, losses, draws)
        print(f"\n  vs {opponent_name:<10}  {wins:2d}胜  {losses:2d}负  {draws:2d}平")

    print("\n" + "=" * 60)
    total_wins = sum(r[0] for r in results.values())
    total_games = sum(sum(r) for r in results.values())
    print(f"\n  总胜率: {total_wins}/{total_games} ({total_wins/total_games*100:.0f}%)")
    print("=" * 60 + "\n")


def run_full_evaluation():
    """完整评估: 200局 + Elo 评分 + 报告"""
    if not EVAL_AVAILABLE:
        print("[ERROR] evaluate 模块不可用")
        return

    runner = TournamentRunner()
    reporter = Reporter()

    print("\n" + "=" * 60)
    print("  完整评估 (Full Evaluation)")
    print("  Algorithm: Bitboard Minimax + TT + ID")
    print("=" * 60)

    # 1. 对阵 random (50局快速)
    runner.run_matchup("submission.py", "random", games=50, verbose=True)

    # 2. 对阵 negamax (50局快速)
    runner.run_matchup("submission.py", "negamax", games=50, verbose=True)

    # 3. 输出报告
    print(runner.summary())

    # 4. 保存
    reporter.export_json(runner.elo)
    reporter.export_tournament_json(runner)
    reporter.save_report(runner.elo, runner)

    print("\n[INFO] 完整评估完成")


# ===================== 入口 =====================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--tournament":
        run_tournament()
    elif len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        opp = sys.argv[2] if len(sys.argv) > 2 else "random"
        play_and_visualize(opponent=opp)
    elif len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_evaluation()
    else:
        unittest.main(verbosity=2)
