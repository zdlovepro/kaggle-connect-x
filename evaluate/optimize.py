"""
超参优化模块 (Optuna)

使用贝叶斯优化自动搜索最优的:
  - 评估函数窗口权重 [w1, w2, w3, w4]
  - 中心列加成
  - 搜索深度分配
  - MCTS 的 C 值和模拟次数

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
    "window_weights": [1.0, 10.0, 100.0, 1000.0],
    "center_bonus": 2.0,
    "depth_early": 4,
    "depth_mid": 5,
    "depth_late": 6,
}

# 位置热图 (6行×7列, 对称)
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
        返回相对于对手的胜率。
        """
        try:
            import optuna
        except ImportError:
            print("[ERROR] optuna not installed. Run: pip install optuna")
            return 0.0

        # ── 采样参数 ──
        # 窗口权重
        w1 = trial.suggest_float("w1", 0.5, 5.0, log=True)
        w2 = trial.suggest_float("w2", 5.0, 50.0, log=True)
        w3 = trial.suggest_float("w3", 50.0, 500.0, log=True)
        w4 = trial.suggest_float("w4", 500.0, 5000.0, log=True)

        # 中心列加成
        center_bonus = trial.suggest_float("center_bonus", 0.0, 20.0)

        # 搜索深度
        depth_early = trial.suggest_int("depth_early", 2, 6)
        depth_mid = trial.suggest_int("depth_mid", 3, 8)
        depth_late = trial.suggest_int("depth_late", 4, 12)

        # MCTS 参数
        c_param = trial.suggest_float("mcts_c", 0.5, 3.0)
        rollouts = trial.suggest_int("mcts_rollouts", 100, 5000, step=100)

        params = {
            "window_weights": [w1, w2, w3, w4],
            "center_bonus": center_bonus,
            "depth_early": depth_early,
            "depth_mid": depth_mid,
            "depth_late": depth_late,
            "mcts_c": c_param,
            "mcts_rollouts": rollouts,
        }

        # ── 构建带参数的 Agent (通过环境变量传递) ──
        os.environ["CX_OPTIMIZE_PARAMS"] = json.dumps(params)

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
            f.write("# 窗口权重: [1-in-row, 2-in-row, 3-in-row, 4-in-row]\n')
            f.write(f"WINDOW_WEIGHTS = {params.get('window_weights', [1, 10, 100, 1000])}\n\n")
            f.write(f"CENTER_BONUS = {params.get('center_bonus', 2.0)}\n\n")
            f.write("# 搜索深度: early / mid / late\n")
            f.write(f"DEPTH_EARLY = {params.get('depth_early', 4)}\n")
            f.write(f"DEPTH_MID = {params.get('depth_mid', 5)}\n")
            f.write(f"DEPTH_LATE = {params.get('depth_late', 6)}\n\n")
            f.write("# MCTS 参数\n")
            f.write(f"MCTS_C = {params.get('mcts_c', 1.414)}\n")
            f.write(f"MCTS_ROLLOUTS = {params.get('mcts_rollouts', 800)}\n")
            f.write(f"MCTS_TIME_BUDGET_MS = 1900\n\n")
            f.write("# 位置热图 (6行×7列, 对称)\n")
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
    parser.add_argument("--trials", type=int, default=200, help="Optuna 试验次数")
    parser.add_argument("--games", type=int, default=50, help="每次试验的对局数")
    parser.add_argument("--opponent", type=str, default="negamax", help="对手")
    parser.add_argument("--timeout", type=int, default=3600, help="总超时 (秒)")

    args = parser.parse_args()

    optimizer = Optimizer(opponent=args.opponent, games_per_trial=args.games)
    best = optimizer.run(n_trials=args.trials, timeout=args.timeout)
