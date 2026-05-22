"""
锦标赛运行器

支持:
  - 多种 Agent 对阵 (通过 Kaggle environments)
  - 可配置局数、先后手
  - 自动 Elo 评分更新
  - 统计汇总输出
"""

import time
from typing import Dict, List, Optional, Tuple

from kaggle_environments import make

from .elo import EloEngine


# ── Agent 包装器 ────────────────────────────────────────────

def wrap_submission_function(submission_path: str) -> str:
    """将 submission.py 中的 agent 函数包装为 Kaggle 可用的 Agent 名"""
    return submission_path


def _agent_factory(agent_spec: str):
    """
    解析 agent 规格字符串。
    支持的格式:
      - "random" / "negamax" → Kaggle 内置
      - "submission.py" → 本地文件
      - "agents.minimax_bitboard" → 模块路径
    """
    # 内置对手
    if agent_spec in ("random", "negamax"):
        return agent_spec

    # 本地文件
    if agent_spec.endswith(".py"):
        return agent_spec

    # 模块路径
    return agent_spec


def run_single_game(agent1_spec: str, agent2_spec: str, debug: bool = False) -> Tuple[int, int]:
    """
    运行一局游戏。

    Returns:
        (reward_p1, reward_p2)
          1 = 胜, 0 = 负/平
    """
    env = make("connectx", debug=debug)
    try:
        env.run([_agent_factory(agent1_spec), _agent_factory(agent2_spec)])
        last_step = env.steps[-1]
        r1 = last_step[0].reward or 0
        r2 = last_step[1].reward or 0
        return int(r1), int(r2)
    except Exception as e:
        print(f"[ERROR] Game failed: {e}")
        return 0, 1  # agent1 判负


# ── 锦标赛运行器 ────────────────────────────────────────────

class TournamentRunner:
    """
    锦标赛运行器。

    用法:
        runner = TournamentRunner(elo=EloEngine())
        runner.run_matchup("submission.py", "random", games=100)
        runner.run_matchup("submission.py", "negamax", games=100)
        print(runner.summary())
    """

    def __init__(self, elo: Optional[EloEngine] = None):
        self.elo = elo or EloEngine()
        self.results: Dict[str, List[Dict]] = {}  # matchup_name → per_game_results

    def run_matchup(
        self,
        agent_a: str,
        agent_b: str,
        games: int = 100,
        verbose: bool = True,
    ) -> Dict:
        """
        运行两个 Agent 之间的多局对战。

        Args:
            agent_a: Agent A 的规格字符串
            agent_b: Agent B 的规格字符串
            games: 总局数 (先后手各半)
            verbose: 是否打印进度

        Returns:
            {wins_a, losses_a, draws_a, win_rate_a, elo_a_before, elo_a_after, ...}
        """
        matchup_name = f"{agent_a} vs {agent_b}"
        if verbose:
            print(f"\n[Tournament] {matchup_name} ({games} games)")

        elo_before_a = self.elo.get_rating(agent_a)
        elo_before_b = self.elo.get_rating(agent_b)

        wins_a = 0
        losses_a = 0
        draws_a = 0
        total_games = 0

        half = games // 2
        if games % 2 != 0:
            half += 1

        for i in range(games):
            # 先后手轮流
            if i < half:
                r1, r2 = run_single_game(agent_a, agent_b)
                # A 先手
                if r1 == 1:
                    wins_a += 1
                elif r2 == 1:
                    losses_a += 1
                else:
                    draws_a += 1
                result_a = 1.0 if r1 == 1 else (0.5 if r1 == 0 else 0.0)
            else:
                r2, r1 = run_single_game(agent_b, agent_a)
                # B 先手，A 后手
                if r1 == 1:
                    wins_a += 1
                elif r2 == 1:
                    losses_a += 1
                else:
                    draws_a += 1
                result_a = 1.0 if r1 == 1 else (0.5 if r1 == 0 else 0.0)

            total_games += 1
            self.elo.update(agent_a, agent_b, result_a)

            if verbose and (i + 1) % max(1, games // 10) == 0:
                wr = wins_a / total_games * 100
                print(f"  Progress: {i + 1}/{games} | Win rate: {wr:.1f}% "
                      f"({wins_a}W/{losses_a}L/{draws_a}D)")

        elo_after_a = self.elo.get_rating(agent_a)
        elo_after_b = self.elo.get_rating(agent_b)

        result = {
            "matchup": matchup_name,
            "games": total_games,
            "wins_a": wins_a,
            "losses_a": losses_a,
            "draws_a": draws_a,
            "win_rate_a": wins_a / total_games if total_games > 0 else 0,
            "elo_a_before": elo_before_a.mu if elo_before_a else None,
            "elo_a_after": elo_after_a.mu if elo_after_a else None,
            "elo_b_before": elo_before_b.mu if elo_before_b else None,
            "elo_b_after": elo_after_b.mu if elo_after_b else None,
            "elo_delta_a": elo_after_a.mu - elo_before_a.mu if elo_before_a and elo_after_a else 0,
        }

        self.results[matchup_name] = result

        if verbose:
            print(f"  Final: {wins_a}W/{losses_a}L/{draws_a}D ({result['win_rate_a']*100:.1f}%) "
                  f"| Elo Δ: {result['elo_delta_a']:+.1f}")

        return result

    def run_round_robin(
        self,
        agents: List[str],
        games_per_matchup: int = 100,
        verbose: bool = True,
    ) -> EloEngine:
        """
        循环赛：每种 Agent 组合都互相打。

        Args:
            agents: Agent 规格列表
            games_per_matchup: 每种配对的局数
            verbose: 是否打印进度

        Returns:
            EloEngine (已更新所有评分)
        """
        n = len(agents)
        for i in range(n):
            for j in range(i + 1, n):
                self.run_matchup(
                    agents[i], agents[j],
                    games=games_per_matchup,
                    verbose=verbose,
                )
        return self.elo

    def summary(self) -> str:
        """生成锦标赛汇总"""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  Tournament Summary")
        lines.append("=" * 70)
        for matchup, result in self.results.items():
            lines.append(
                f"  {matchup:<40} "
                f"{result['wins_a']:>3}W/{result['losses_a']:>3}L/{result['draws_a']:>3}D  "
                f"WR: {result['win_rate_a']*100:>5.1f}%  "
                f"EloΔ: {result['elo_delta_a']:>+6.1f}"
            )
        lines.append("=" * 70)
        lines.append(self.elo.summary())
        return "\n".join(lines)


# ── 性能基准测试 ────────────────────────────────────────────

class BenchmarkRunner:
    """
    性能基准测试运行器。

    测量每个 Agent 的:
      - 每步耗时 (min/avg/max/p99)
      - 搜索节点数
      - 剪枝效率
    """

    def __init__(self):
        self.benchmarks: Dict[str, Dict] = {}

    def benchmark_agent(
        self,
        agent_wrapper,
        board_configs: List,
        repetitions: int = 5,
    ) -> Dict:
        """
        对单个 Agent 做性能测试。

        Args:
            agent_wrapper: 一个可调用对象，接受 (observation, configuration)
            board_configs: 测试用的不同棋盘状态列表
            repetitions: 每种配置重复次数

        Returns:
            {time_ms_avg, time_ms_min, time_ms_max, time_ms_p99, nodes_visited, ...}
        """
        import statistics

        times = []
        nodes_list = []

        for board in board_configs:
            for _ in range(repetitions):
                start = time.perf_counter()
                action = agent_wrapper(board)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

                # 尝试获取节点统计 (如果 agent 支持)
                if hasattr(agent_wrapper, 'stats'):
                    nodes_list.append(agent_wrapper.stats.get("nodes_visited", 0))

        times_sorted = sorted(times)
        p99_idx = int(len(times_sorted) * 0.99)

        result = {
            "samples": len(times),
            "time_ms_avg": statistics.mean(times),
            "time_ms_median": statistics.median(times),
            "time_ms_min": min(times),
            "time_ms_max": max(times),
            "time_ms_p99": times_sorted[p99_idx] if times_sorted else 0,
            "time_ms_stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "nodes_avg": statistics.mean(nodes_list) if nodes_list else 0,
        }

        self.benchmarks[str(id(agent_wrapper))] = result
        return result

    def compare(self, agent_results: Dict[str, Dict]) -> str:
        """生成性能对比报告"""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  Performance Benchmark Comparison")
        lines.append("=" * 70)
        lines.append(
            f"{'Agent':<30} {'Avg(ms)':>8} {'Median':>8} "
            f"{'Min':>8} {'Max':>8} {'P99':>8}"
        )
        lines.append("-" * 70)
        for name, result in agent_results.items():
            lines.append(
                f"{name:<30} {result['time_ms_avg']:>8.1f} "
                f"{result['time_ms_median']:>8.1f} {result['time_ms_min']:>8.1f} "
                f"{result['time_ms_max']:>8.1f} {result['time_ms_p99']:>8.1f}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)
