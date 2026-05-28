"""
超参优化模块 (Optuna)

使用贝叶斯优化自动搜索最优的:
  - 评估函数权重 (2-in-a-row, 3-in-a-row)
  - 位置热图倍率
  - 奇偶威胁倍率

运行方式:
  python evaluate/optimize.py --trials 200

注意: 需要在有 Optuna 的环境中运行，结果写入 agents/weights.py
"""

import os
import sys
import json
import math
from typing import Dict, Any, List

# 将项目根目录加入 Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.tournament import run_single_game, TournamentRunner
from evaluate.elo import EloEngine


# ── 评估权重配置 ─────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "w2": 10.0,
    "w3": 100.0,
    "w_hmap": 0.3,
    "w_odd_even": 80.0,
}

# 位置热图 (6行x7列, 对称)
DEFAULT_POSITION_HEATMAP = [
    [3, 4, 5, 7, 5, 4, 3],
    [4, 6, 8, 10, 8, 6, 4],
    [5, 8, 11, 13, 11, 8, 5],
    [5, 8, 11, 13, 11, 8, 5],
    [4, 6, 8, 10, 8, 6, 4],
    [3, 4, 5, 7, 5, 4, 3],
]


class Optimizer:
    """超参优化器"""

    def __init__(self, opponent: str = "negamax", games_per_trial: int = 50):
        self.opponent = opponent
        self.games_per_trial = games_per_trial
        self.best_params = None
        self.best_score = -float("inf")
        self.history: List[Dict] = []

    def objective(self, trial) -> float:
        """
        Optuna 目标函数。
        通过 monkey-patch submission.py 全局变量注入评估参数后运行对战。
        """
        try:
            import optuna
        except ImportError:
            print("[ERROR] optuna not installed. Run: pip install optuna")
            return 0.0

        import submission as sub

        # ── 采样参数 ──
        w2 = trial.suggest_float("w2", 5.0, 50.0, log=True)
        w3 = trial.suggest_float("w3", 50.0, 500.0, log=True)
        w_hmap = trial.suggest_float("w_hmap", 0.05, 2.0, log=True)
        w_odd_even = trial.suggest_float("w_odd_even", 20.0, 300.0, log=True)

        params = {
            "w2": w2, "w3": w3,
            "w_hmap": w_hmap, "w_odd_even": w_odd_even,
        }

        # ── 注入参数到 submission.py (monkey-patch) ──
        old_w_score = sub._W_SCORE
        old_w_threat = sub._W_THREAT
        old_w_hmap = sub._W_HMAP
        old_w_odd_even = sub._W_ODD_EVEN
        sub._W_SCORE = float(w2)
        sub._W_THREAT = float(w3)
        sub._W_HMAP = float(w_hmap)
        sub._W_ODD_EVEN = float(w_odd_even)

        # ── 运行对战 ──
        wins = 0
        losses = 0
        draws = 0

        for i in range(self.games_per_trial):
            if i < self.games_per_trial // 2:
                r1, r2 = run_single_game("submission.py", self.opponent)
            else:
                r2, r1 = run_single_game(self.opponent, "submission.py")

            if r1 == 1:
                wins += 1
            elif r2 == 1:
                losses += 1
            else:
                draws += 1

        win_rate = wins / self.games_per_trial if self.games_per_trial > 0 else 0

        # ── 恢复原始值 ──
        sub._W_SCORE = old_w_score
        sub._W_THREAT = old_w_threat
        sub._W_HMAP = old_w_hmap
        sub._W_ODD_EVEN = old_w_odd_even

        # 记录
        self.history.append({
            "params": params,
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "draws": draws,
        })

        if win_rate > self.best_score:
            self.best_score = win_rate
            self.best_params = params

        return win_rate

    def run(self, n_trials: int = 200, timeout: int = 3600):
        """运行超参优化"""
        try:
            import optuna
        except ImportError:
            print("[ERROR] Optuna not installed. Install with: pip install optuna")
            print("[INFO] Using default parameters instead.")
            return

        study = optuna.create_study(
            direction="maximize",
            study_name="connectx_optimization",
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        print("\n" + "=" * 60)
        print("  Optimization Complete")
        print("=" * 60)
        print(f"  Best Score: {study.best_value:.3f}")
        print(f"  Best Params: {study.best_params}")
        print("=" * 60)

        # 保存最优权重
        self._save_best_weights(study.best_params)

        return study.best_params

    def _save_best_weights(self, params: Dict):
        """将最优权重保存为 Python 模块"""
        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "weights.py"
        )

        with open(weights_path, "w", encoding="utf-8") as f:
            f.write('"""自动优化的评估权重 (由 evaluate/optimize.py 生成)"""\n\n')
            f.write("# 评估权重: 2-in-a-row, 3-in-a-row\n")
            f.write(f"W_SCORE = {params.get('w2', 10.0)}\n")
            f.write(f"W_THREAT = {params.get('w3', 100.0)}\n\n")
            f.write("# 位置热图倍率\n")
            f.write(f"W_HMAP = {params.get('w_hmap', 0.3)}\n\n")
            f.write("# 奇偶威胁倍率\n")
            f.write(f"W_ODD_EVEN = {params.get('w_odd_even', 80.0)}\n\n")
            f.write("# 位置热图 (6行x7列, 对称)\n")
            f.write("POSITION_HEATMAP = [\n")
            f.write("    [3, 4, 5, 7, 5, 4, 3],\n")
            f.write("    [4, 6, 8, 10, 8, 6, 4],\n")
            f.write("    [5, 8, 11, 13, 11, 8, 5],\n")
            f.write("    [5, 8, 11, 13, 11, 8, 5],\n")
            f.write("    [4, 6, 8, 10, 8, 6, 4],\n")
            f.write("    [3, 4, 5, 7, 5, 4, 3],\n")
            f.write("]\n")

        print(f"[INFO] Best weights saved to {weights_path}")


# ── 独立入口 ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ConnectX Hyperparameter Optimization")
    parser.add_argument("--trials", type=int, default=200, help="Optuna trial count")
    parser.add_argument("--games", type=int, default=50, help="Games per trial")
    parser.add_argument("--opponent", type=str, default="negamax", help="Opponent agent")
    parser.add_argument("--timeout", type=int, default=3600, help="Total timeout (seconds)")

    args = parser.parse_args()

    optimizer = Optimizer(opponent=args.opponent, games_per_trial=args.games)
    best = optimizer.run(n_trials=args.trials, timeout=args.timeout)
