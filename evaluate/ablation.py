"""
消融实验模块

测试移除单个优化组件后 Agent 表现的下降幅度，量化每个组件贡献。

支持的一对一消融:
  - 有着法排序 vs 无着法排序 (random order)
  - 有 Alpha-Beta 剪枝 vs 纯 Minimax
  - Bitboard 表示 vs 列表表示
  - 有置换表 vs 无置换表
  - 深搜索 vs 浅搜索
"""

from typing import Dict, List, Optional

from .tournament import TournamentRunner, run_single_game


# ── 消融变体工厂 ────────────────────────────────────────────

def _make_ablation_variant(base_agent_path: str, disable: str) -> str:
    """
    生成消融变体。

    由于 kaggle_environments 的限制，消融测试使用不同的
    submission.py 副本（带开关参数）。
    """
    return f"{base_agent_path}#{disable}"


def run_ablation(
    agent_a: str,
    agent_b: str,
    games: int = 50,
    label: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    运行消融对比实验。

    Args:
        agent_a: 完整 Agent
        agent_b: 消融变体
        games: 局数
        label: 实验标签
        verbose: 是否打印进度

    Returns:
        {
            label, games, wins_a, losses_a, draws_a,
            win_rate_a, win_rate_b, delta_percentage
        }
    """
    if label is None:
        label = f"{agent_a} - {agent_b}"

    if verbose:
        print(f"\n[Ablation] {label} ({games} games)")

    wins_a = 0
    losses_a = 0
    draws_a = 0

    for i in range(games):
        if i < games // 2:
            r1, r2 = run_single_game(agent_a, agent_b)
        else:
            r2, r1 = run_single_game(agent_b, agent_a)

        if r1 == 1:
            wins_a += 1
        elif r2 == 1:
            losses_a += 1
        else:
            draws_a += 1

        if verbose and (i + 1) % max(1, games // 5) == 0:
            wr = wins_a / (i + 1) * 100
            print(f"  {i + 1}/{games} | WR: {wr:.1f}%")

    win_rate_a = wins_a / games
    delta = win_rate_a - 0.5  # 相对 50% 的偏差

    result = {
        "label": label,
        "games": games,
        "wins_full": wins_a,
        "losses_full": losses_a,
        "draws": draws_a,
        "win_rate_full": win_rate_a,
        "win_rate_ablation": 1 - win_rate_a,
        "delta": delta,
        "delta_percentage": delta * 100,
    }

    if verbose:
        print(f"  Result: Full={win_rate_a*100:.1f}% Ablation={(1-win_rate_a)*100:.1f}% "
              f"Δ={delta*100:+.1f}%")

    return result


def ablation_suite(
    base_agent: str = "submission.py",
    games: int = 50,
    verbose: bool = True,
) -> List[Dict]:
    """
    运行完整消融测试套件。

    Returns:
        各项消融实验结果列表
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  Ablation Study Suite")
        print("=" * 60)

    results = []

    # 这里演示典型的消融项目
    # 由于 kaggle_environments 每次都会重新加载 agent，
    # 实际的消融测试需要不同配置的 submission.py 副本来实现。
    #
    # 以下为示例结构，实际使用时需要配合不同配置的 agent 文件。

    experiments = [
        # (agent_a, agent_b, label, games)
        # 实际使用时替换为不同配置的 agent 路径
    ]

    for agent_a, agent_b, label, n_games in experiments:
        result = run_ablation(agent_a, agent_b, games=n_games, label=label, verbose=verbose)
        results.append(result)

    if verbose:
        print("\n" + "-" * 60)
        print("  Ablation Summary:")
        for r in results:
            print(f"  {r['label']:<40} Δ = {r['delta_percentage']:>+5.1f}%")
        print("=" * 60)

    return results
