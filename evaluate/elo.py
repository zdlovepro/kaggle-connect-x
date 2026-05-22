"""
Elo 评分引擎

实现标准 Elo 评分算法，用于量化不同 Agent 版本的相对实力。

Elo 公式:
  E_A = 1 / (1 + 10^((μ_B - μ_A) / 400))
  μ'_A = μ_A + K * (S_A - E_A)

其中:
  - μ: 技能评分 (初始 1000)
  - K: 更新因子 (默认 32)
  - S: 实际得分 (1=胜, 0.5=平, 0=负)
  - E: 预期得分
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class EloRating:
    """单个 Agent 的 Elo 评分状态"""
    name: str
    mu: float = 1000.0       # 技能评分
    sigma: float = 350.0     # 不确定性
    games: int = 0           # 总对局数
    wins: int = 0
    losses: int = 0
    draws: int = 0
    history: List[Tuple[int, float, float]] = field(default_factory=list)  # (games, mu, sigma)


class EloEngine:
    """Elo 评分引擎"""

    BASE_K = 32
    INITIAL_SIGMA = 350.0
    MIN_SIGMA = 50.0

    def __init__(self, initial_mu: float = 1000.0, k_factor: float = 32):
        self.initial_mu = initial_mu
        self.k_factor = k_factor
        self.ratings: Dict[str, EloRating] = {}

    def register(self, name: str, initial_mu: Optional[float] = None) -> EloRating:
        """注册一个新 Agent"""
        if name not in self.ratings:
            rating = EloRating(
                name=name,
                mu=initial_mu if initial_mu is not None else self.initial_mu,
                sigma=self.INITIAL_SIGMA,
            )
            self.ratings[name] = rating
        return self.ratings[name]

    def get_rating(self, name: str) -> Optional[EloRating]:
        return self.ratings.get(name)

    def expected_score(self, rating_a: float, rating_b: float) -> Tuple[float, float]:
        """计算预期得分 E_A, E_B"""
        e_a = 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))
        e_b = 1.0 - e_a
        return e_a, e_b

    def update(self, name_a: str, name_b: str, result_a: float):
        """
        根据一局结果更新评分。

        Args:
            name_a: Agent A 的名称
            name_b: Agent B 的名称
            result_a: A 的得分 (1.0=胜, 0.5=平, 0.0=负)
        """
        r_a = self.register(name_a)
        r_b = self.register(name_b)

        # 预期得分
        e_a, _ = self.expected_score(r_a.mu, r_b.mu)

        # K 因子动态调整 (不确定性越大，变化越大)
        k_a = self.k_factor * (r_a.sigma / self.INITIAL_SIGMA)
        k_b = self.k_factor * (r_b.sigma / self.INITIAL_SIGMA)

        # 更新
        r_a.mu += k_a * (result_a - e_a)
        r_b.mu += k_b * ((1.0 - result_a) - (1.0 - e_a))

        # 不确定性衰减
        r_a.sigma = max(self.MIN_SIGMA, r_a.sigma * 0.99)
        r_b.sigma = max(self.MIN_SIGMA, r_b.sigma * 0.99)

        # 统计
        r_a.games += 1
        r_b.games += 1
        if result_a == 1.0:
            r_a.wins += 1
            r_b.losses += 1
        elif result_a == 0.0:
            r_a.losses += 1
            r_b.wins += 1
        else:
            r_a.draws += 1
            r_b.draws += 1

    def update_batch(self, name_a: str, name_b: str, results: List[float]):
        """批量更新 (多局结果)"""
        for result in results:
            self.update(name_a, name_b, result)

    def get_confidence_interval(self, name: str, z: float = 1.96) -> Tuple[float, float]:
        """获取 95% 置信区间 [μ - z*σ, μ + z*σ]"""
        r = self.ratings[name]
        return r.mu - z * r.sigma, r.mu + z * r.sigma

    def win_rate(self, name: str) -> Optional[float]:
        """胜率"""
        r = self.ratings.get(name)
        if r is None or r.games == 0:
            return None
        return r.wins / r.games

    def ranking(self) -> List[EloRating]:
        """按 μ 降序排列"""
        return sorted(self.ratings.values(), key=lambda r: r.mu, reverse=True)

    def summary(self) -> str:
        """生成评分汇总"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"{'Agent':<30} {'μ (Elo)':>8} {'±95%CI':>12} {'胜/负/平':>20} {'胜率':>8}")
        lines.append("-" * 70)
        for r in self.ranking():
            lo, hi = self.get_confidence_interval(r.name)
            ci = f"±{r.mu - lo:.0f}"
            record = f"{r.wins}/{r.losses}/{r.draws}"
            wr = f"{r.wins/r.games*100:.1f}%" if r.games > 0 else "N/A"
            lines.append(
                f"{r.name:<30} {r.mu:>8.1f} {ci:>12} {record:>20} {wr:>8}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """导出为字典"""
        return {
            name: {
                "mu": r.mu, "sigma": r.sigma, "games": r.games,
                "wins": r.wins, "losses": r.losses, "draws": r.draws,
                "history": r.history,
            }
            for name, r in self.ratings.items()
        }

    def save_snapshot(self, name: str):
        """保存当前评分快照"""
        r = self.ratings.get(name)
        if r:
            r.history.append((r.games, r.mu, r.sigma))
