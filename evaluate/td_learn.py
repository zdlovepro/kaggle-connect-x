"""
TD-Learning 评估函数参数优化

使用 TD(0) 算法通过自对弈学习最优评估函数权重。
保持 NegaScout 搜索框架不变，仅优化评估函数参数。

方法:
  线性值函数近似: V(s) = tanh(w · f(s))
  TD(0) 更新: w ← w + α * δ * f(s)

  其中 δ = target - V(s_t):
    - 非终端: target = -V(s_{t+1})  (视角切换)
    - 终端:   target = outcome  (±1 = 胜/负, 0 = 平)

训练流程:
  1. 自对弈 (ε-greedy, 基于当前权重的 1-ply 贪心)
  2. TD(0) 权重更新 (可选 TD(λ) eligibility traces)
  3. 周期性评估 (vs random / negamax)
  4. 保存最优权重到 agents/td_weights.py

断点续训:
  --resume  从 agents/td_weights.py 恢复权重继续训练
  --load PATH  从指定文件恢复权重继续训练
  每轮 eval 结束后自动保存 checkpoint (含 episode/history)，中断后可恢复。
"""

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Ensure project root is importable when running `python evaluate/td_learn.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectx.bitboard import (
    COLS, ROWS, INAROW, ROW_STRIDE,
    has_won, list_to_bitboards,
    get_heights_from_list, get_heights, valid_cols, drop_pos,
    HEATMAP, HMAP_POS, COL0_MASK, COL6_MASK,
)
try:
    from evaluate.build_submission import (
        BuildConfig,
        build_submission,
        build_submission_with_config,
    )
except Exception:
    from build_submission import BuildConfig, build_submission, build_submission_with_config


# ── 特征提取 ─────────────────────────────────────────────────

def extract_features(b_self: np.uint64, b_opp: np.uint64,
                     heights: np.ndarray) -> np.ndarray:
    """
    从 bitboard 提取 11 维特征向量。

    [0]  bias (1.0)
    [1]  self 2-in-a-row
    [2]  self 3-in-a-row
    [3]  opp 2-in-a-row
    [4]  opp 3-in-a-row
    [5]  self heatmap score
    [6]  opp heatmap score
    [7]  self odd-row threat
    [8]  self even-row threat
    [9]  opp odd-row threat
    [10] opp even-row threat
    """
    features = np.zeros(11, dtype=np.float32)
    features[0] = 1.0

    for idx, b in enumerate([b_self, b_opp]):
        base = 1 if idx == 0 else 3
        b_nor = b & ~COL6_MASK
        b_nol = b & ~COL0_MASK

        h2 = b_nor & (b >> np.uint64(1))
        v2 = b & (b >> np.uint64(ROW_STRIDE))
        d12 = b_nor & (b >> np.uint64(ROW_STRIDE + 1))
        d22 = b_nol & (b >> np.uint64(ROW_STRIDE - 1))
        c2 = h2.bit_count() + v2.bit_count() + d12.bit_count() + d22.bit_count()
        features[base] = float(c2)

        h3 = h2 & (b >> np.uint64(2))
        v3 = v2 & (b >> np.uint64(ROW_STRIDE))
        d13 = d12 & (b >> np.uint64(ROW_STRIDE + 2))
        d23 = d22 & (b >> np.uint64(ROW_STRIDE - 2))
        c3 = h3.bit_count() + v3.bit_count() + d13.bit_count() + d23.bit_count()
        features[base + 1] = float(c3)

    # heatmap
    for idx, b in enumerate([b_self, b_opp]):
        fb = 5 if idx == 0 else 6
        bb = b
        while bb:
            lsb = int(bb & (~bb + np.uint64(1)))
            pos = lsb.bit_length() - 1
            if pos < len(HMAP_POS):
                features[fb] += HMAP_POS[pos]
            bb ^= lsb

    # odd/even threats
    filled = sum(int(heights[c]) for c in range(COLS))
    phase = filled / (COLS * ROWS)
    scale = phase

    for c in range(COLS):
        h = int(heights[c])
        if h < 3:
            continue
        complete_row = ROWS - 1 - h
        bottom_parity = h % 2

        all_self = True
        all_opp = True
        for r in range(complete_row + 1, complete_row + 4):
            mask = np.uint64(1) << np.uint64(r * ROW_STRIDE + c)
            if not (b_self & mask):
                all_self = False
            if not (b_opp & mask):
                all_opp = False

        if all_self:
            if bottom_parity == 1:
                features[7] += 2.0 * scale + 0.125
            else:
                features[8] += 0.5 * scale + 0.0375
        elif all_opp:
            if bottom_parity == 1:
                features[9] += 2.0 * scale + 0.125
            else:
                features[10] += 0.5 * scale + 0.0375

    return features


# ── 值函数 ────────────────────────────────────────────────────

def predict_value(features: np.ndarray, weights: np.ndarray) -> float:
    """V(s) = tanh(w · f)  →  [-1, 1]"""
    raw = float(np.dot(weights, features))
    return math.tanh(raw)


# ── 自对弈 ────────────────────────────────────────────────────

def _select_greedy(b_self: np.uint64, b_opp: np.uint64,
                   heights: np.ndarray, weights: np.ndarray,
                   epsilon: float) -> int:
    """ε-greedy: 选择最小化对手价值的着法 (argmin_c V(s'))"""
    valid = valid_cols(heights)
    if random.random() < epsilon:
        return random.choice(valid)

    best_col = valid[0]
    best_val = float('inf')
    for col in valid:
        pos = drop_pos(heights, col)
        mask = np.uint64(1) << np.uint64(pos)
        heights[col] += 1
        feat = extract_features(b_self | mask, b_opp, heights)
        val = predict_value(feat, weights)
        heights[col] -= 1
        if val < best_val:
            best_val = val
            best_col = col
    return best_col


def play_self_play_game(weights: np.ndarray,
                        epsilon: float = 0.1) -> List[dict]:
    """
    运行一局自对弈，返回轨迹列表。
    每步: {features, player, outcome} (outcome 在游戏结束后回填)
    """
    b1 = np.uint64(0)
    b2 = np.uint64(0)
    heights = np.zeros(COLS, dtype=np.int32)
    player = 1
    trajectory = []

    while True:
        valid = valid_cols(heights)
        if not valid:
            break

        b_self = b1 if player == 1 else b2
        b_opp = b2 if player == 1 else b1
        feat = extract_features(b_self, b_opp, heights).copy()

        # 立即获胜
        for col in valid:
            pos = drop_pos(heights, col)
            current = b_self
            new_b = current | (np.uint64(1) << np.uint64(pos))
            if has_won(new_b):
                if player == 1:
                    b1 |= (np.uint64(1) << np.uint64(pos))
                else:
                    b2 |= (np.uint64(1) << np.uint64(pos))
                trajectory.append({'features': feat, 'player': player})
                for i in range(len(trajectory) - 1, -1, -1):
                    p = trajectory[i]['player']
                    trajectory[i]['outcome'] = 1.0 if p == player else -1.0
                return trajectory

        # 拦截对手
        opp_board = b_opp
        blocked = False
        for col in valid:
            pos = drop_pos(heights, col)
            if has_won(opp_board | (np.uint64(1) << np.uint64(pos))):
                heights[col] += 1
                if player == 1:
                    b1 |= (np.uint64(1) << np.uint64(pos))
                else:
                    b2 |= (np.uint64(1) << np.uint64(pos))
                trajectory.append({'features': feat, 'player': player})
                player = 3 - player
                blocked = True
                break
        if blocked:
            continue

        # ε-greedy
        col = _select_greedy(b_self, b_opp, heights, weights, epsilon)
        pos = drop_pos(heights, col)
        mask = np.uint64(1) << np.uint64(pos)
        heights[col] += 1
        if player == 1:
            b1 |= mask
        else:
            b2 |= mask

        trajectory.append({'features': feat, 'player': player})

        if has_won(b1 if player == 1 else b2):
            for i in range(len(trajectory) - 1, -1, -1):
                p = trajectory[i]['player']
                trajectory[i]['outcome'] = 1.0 if p == player else -1.0
            return trajectory

        player = 3 - player

    for step in trajectory:
        step['outcome'] = 0.0
    return trajectory


# ── TD 更新 ───────────────────────────────────────────────────

def td_update(weights: np.ndarray, trajectory: List[dict],
              alpha: float, lam: float = 0.0) -> np.ndarray:
    """TD(λ) 权重更新。lam=0 即 TD(0)。"""
    n = len(trajectory)
    if n == 0:
        return weights

    eligibility = np.zeros_like(weights)

    for t in range(n - 1, -1, -1):
        feat_t = trajectory[t]['features']
        v_t = predict_value(feat_t, weights)

        if t == n - 1:
            target = trajectory[t]['outcome']
        else:
            target = -predict_value(trajectory[t + 1]['features'], weights)

        td_err = target - v_t

        if lam > 0:
            eligibility = lam * eligibility + feat_t
            weights += alpha * td_err * eligibility
        else:
            weights += alpha * td_err * feat_t

    return weights


# ── 评估 (通过 kaggle_environments) ───────────────────────────

def _build_td_eval_submission(weights: np.ndarray, output_path: Path) -> str:
    """Build a temporary single-file submission using TD feature weights."""
    root = Path(__file__).resolve().parents[1]
    cfg = BuildConfig(
        w_score=float(max(0.1, abs(weights[1]) * 170.0)),
        w_threat=float(max(1.0, abs(weights[2]) * 170.0)),
        w_hmap=0.3,
        w_odd_even=80.0,
        heatmap=HEATMAP.astype(float).tolist(),
        td_weights=[float(v) for v in weights],
        source="td_learn.evaluate_against",
    )
    build_submission_with_config(root / "submission.py", output_path, cfg)
    return str(output_path)


def evaluate_against(weights: np.ndarray, opponent: str = "random",
                     num_games: int = 100) -> dict:
    """Evaluate TD weights via the same built single-file path used for submission."""
    from kaggle_environments import make

    temp_submission = Path(__file__).with_name(
        f"_tmp_td_eval_submission_{os.getpid()}.py"
    )
    agent_path = _build_td_eval_submission(weights, temp_submission)

    wins = losses = draws = 0
    half = num_games // 2

    for i in range(num_games):
        if i < half:
            env = make("connectx", debug=False)
            env.run([agent_path, opponent])
            r0 = env.steps[-1][0].reward or 0
            r1 = env.steps[-1][1].reward or 0
            if r0 == 1:
                wins += 1
            elif r1 == 1:
                losses += 1
            else:
                draws += 1
        else:
            env = make("connectx", debug=False)
            env.run([opponent, agent_path])
            r0 = env.steps[-1][0].reward or 0
            r1 = env.steps[-1][1].reward or 0
            if r1 == 1:
                wins += 1
            elif r0 == 1:
                losses += 1
            else:
                draws += 1

    if temp_submission.exists():
        temp_submission.unlink()

    return {'wins': wins, 'losses': losses, 'draws': draws,
            'win_rate': wins / num_games, 'games': num_games}


def _normalize_eval_mix(eval_mix: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Normalize opponent weights to sum to 1.0 and drop non-positive entries."""
    cleaned: List[Tuple[str, float]] = []
    for opp, weight in eval_mix:
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if w > 0:
            cleaned.append((str(opp), w))
    if not cleaned:
        return [("random", 1.0)]
    s = sum(w for _, w in cleaned)
    if s <= 0:
        return [("random", 1.0)]
    return [(opp, w / s) for opp, w in cleaned]


def _allocate_eval_games(total_games: int, eval_mix: List[Tuple[str, float]]) -> Dict[str, int]:
    """
    Allocate total evaluation games to each opponent by normalized weight.
    Uses largest-remainder rounding to keep total exactly total_games.
    """
    if total_games <= 0:
        return {}

    mix = _normalize_eval_mix(eval_mix)
    ideal = [(opp, total_games * w) for opp, w in mix]
    base = {opp: int(v) for opp, v in ideal}
    used = sum(base.values())
    remain = total_games - used

    if remain > 0:
        order = sorted(ideal, key=lambda x: (x[1] - int(x[1])), reverse=True)
        i = 0
        while remain > 0 and order:
            opp = order[i % len(order)][0]
            base[opp] += 1
            remain -= 1
            i += 1

    # Ensure each allocated bucket has enough games for seat-swap fairness.
    # If a bucket gets 1 game, move one game from the largest bucket.
    ones = [opp for opp, g in base.items() if g == 1]
    for opp in ones:
        donor = max(base.items(), key=lambda kv: kv[1])[0]
        if donor != opp and base[donor] >= 3:
            base[donor] -= 1
            base[opp] += 1

    # Prune zeros.
    return {opp: g for opp, g in base.items() if g > 0}


def evaluate_mixed(weights: np.ndarray,
                   eval_mix: List[Tuple[str, float]],
                   total_games: int = 100) -> dict:
    """
    Evaluate weights against multiple opponents and return a weighted score.
    score = sum(weight_i * win_rate_i)
    """
    mix = _normalize_eval_mix(eval_mix)
    plan = _allocate_eval_games(total_games, mix)
    if not plan:
        return {"score": 0.0, "details": {}}

    weight_map = {opp: w for opp, w in mix}
    details: Dict[str, dict] = {}
    score = 0.0
    norm = 0.0

    for opp, games in plan.items():
        result = evaluate_against(weights, opponent=opp, num_games=games)
        details[opp] = result
        w = weight_map.get(opp, 0.0)
        score += w * float(result["win_rate"])
        norm += w

    if norm <= 0.0:
        return {"score": 0.0, "details": details}
    return {"score": score / norm, "details": details}


# ── 断点加载 ──────────────────────────────────────────────────

def load_td_weights(filepath: str) -> np.ndarray:
    """从 agents/td_weights.py 格式文件加载权重向量。"""
    ns = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        exec(f.read(), ns)
    weights = np.array(ns['TD_WEIGHTS'], dtype=np.float64)
    if weights.shape != (11,):
        raise ValueError(f"Expected 11 weights, got {weights.shape[0]}")
    return weights


_CHECKPOINT_SUFFIX = '.checkpoint.json'


def _checkpoint_path(weights_file: str) -> str:
    base, _ = os.path.splitext(weights_file)
    return base + _CHECKPOINT_SUFFIX


def load_checkpoint(weights_file: str) -> dict:
    """从 JSON checkpoint 加载完整训练状态。若 checkpoint 不存在则仅加载权重。"""
    cp_path = _checkpoint_path(weights_file)
    if os.path.exists(cp_path):
        with open(cp_path, 'r', encoding='utf-8') as f:
            cp = json.load(f)
        return {
            'weights': np.array(cp['weights'], dtype=np.float64),
            'episode': cp['episode'],
            'best_win_rate': cp['best_win_rate'],
            'best_weights': np.array(cp['best_weights'], dtype=np.float64),
            'history': [(int(e), float(r)) for e, r in cp.get('history', [])],
        }
    # fallback: 仅加载权重，从零开始
    return {
        'weights': load_td_weights(weights_file),
        'episode': 0,
        'best_win_rate': -1.0,
        'best_weights': None,
        'history': [],
    }


def save_checkpoint(weights: np.ndarray, best_weights: np.ndarray,
                    episode: int, best_win_rate: float,
                    history: list, weights_file: str):
    """保存 JSON checkpoint 供断点续训。"""
    cp = {
        'weights': weights.tolist(),
        'best_weights': best_weights.tolist(),
        'episode': episode,
        'best_win_rate': best_win_rate,
        'history': [[int(e), round(float(r), 6)] for e, r in history],
    }
    cp_path = _checkpoint_path(weights_file)
    with open(cp_path, 'w', encoding='utf-8') as f:
        json.dump(cp, f, indent=2)
    # 不在 verbose 输出中打印，由调用者决定


# ── 权重导出 ──────────────────────────────────────────────────

def export_weights(weights: np.ndarray, filepath: str):
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('"""TD-Learning 学到的评估函数权重 (自动生成)"""\n\n')
        f.write('# 特征权重向量 [bias, w_2self, w_3self, w_2opp, w_3opp,\n')
        f.write('#              w_hmap_self, w_hmap_opp,\n')
        f.write('#              w_odd_self, w_even_self, w_odd_opp, w_even_opp]\n')
        f.write(f'TD_WEIGHTS = {weights.tolist()}\n\n')
        w2 = float(max(0.1, abs(weights[1]) * 170))
        w3 = float(max(1.0, abs(weights[2]) * 170))
        f.write(f'W_SCORE = {w2}\n')
        f.write(f'W_THREAT = {w3}\n')
    print(f"[TD Learn] Weights exported to {filepath}")


# ── 训练器 ────────────────────────────────────────────────────

class TDTrainer:
    """TD-Learning 训练器"""

    _DEFAULT_WEIGHTS = np.array([
        0.0,       # bias
        0.06,      # w_2self   (10/170 ≈ 0.06)
        0.6,       # w_3self   (100/170 ≈ 0.6)
        -0.06,     # w_2opp
        -0.6,      # w_3opp
        0.0018,    # w_hmap_self  (0.3/170 ≈ 0.0018)
        -0.0018,   # w_hmap_opp
        0.06,      # w_odd_self
        0.015,     # w_even_self
        -0.06,     # w_odd_opp
        -0.015,    # w_even_opp
    ], dtype=np.float64)

    def __init__(self, weights: Optional[np.ndarray] = None,
                 episode: int = 0, best_weights: Optional[np.ndarray] = None,
                 best_win_rate: float = -1.0,
                 history: Optional[List[Tuple[int, float]]] = None):
        """
        weights: 当前权重 (None = 使用默认初始值)
        episode: 已完成的训练局数
        best_weights: 历史最优权重 (None = 使用 weights)
        best_win_rate: 历史最优胜率
        history: [(episode, win_rate), ...]
        """
        self.weights = (weights.copy() if weights is not None
                        else self._DEFAULT_WEIGHTS.copy())
        self.best_weights = (best_weights.copy() if best_weights is not None
                             else self.weights.copy())
        self.best_win_rate = best_win_rate
        self.episode = episode
        self.history = list(history) if history else []

    @classmethod
    def from_checkpoint(cls, weights_file: str) -> "TDTrainer":
        """从 checkpoint 恢复训练器 (若存在)，否则从权重文件加载。"""
        cp = load_checkpoint(weights_file)
        best = cp['best_weights']
        if best is None:
            best = cp['weights'].copy()
        return cls(
            weights=cp['weights'],
            episode=cp['episode'],
            best_weights=best,
            best_win_rate=cp['best_win_rate'],
            history=cp['history'],
        )

    def train(self, num_episodes: int = 50000, alpha: float = 0.005,
              lam: float = 0.3, epsilon_start: float = 0.3,
              epsilon_end: float = 0.02, eval_interval: int = 5000,
              eval_games: int = 50, verbose: bool = True,
              alpha_decay: float = 0.9999,
              checkpoint_path: Optional[str] = None,
              eval_mix: Optional[List[Tuple[str, float]]] = None):
        """
        alpha_decay: per-episode learning rate multiplier (1.0=no decay).
        checkpoint_path: 每轮 eval 后自动保存 checkpoint (None=不保存).
        eval_mix: [(opponent, weight), ...] for model selection.
        """
        start_time = time.perf_counter()
        start_ep = self.episode
        lr = alpha
        mix = _normalize_eval_mix(eval_mix or [("random", 0.35), ("negamax", 0.65)])
        eval_count = 0
        if self.episode > 0:
            lr = alpha * (alpha_decay ** self.episode)

        for ep in range(num_episodes):
            progress = (self.episode + ep) / max(1, self.episode + num_episodes - 1)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
            lr = max(alpha * 0.01, lr * alpha_decay)

            trajectory = play_self_play_game(self.weights, epsilon)
            self.weights = td_update(self.weights, trajectory, lr, lam)
            self.episode += 1

            if eval_games > 0 and eval_interval > 0 and (ep + 1) % eval_interval == 0:
                eval_count += 1
                eval_result = evaluate_mixed(
                    self.weights,
                    eval_mix=mix,
                    total_games=eval_games,
                )
                wr = float(eval_result['score'])
                elapsed = time.perf_counter() - start_time

                if verbose:
                    details = eval_result.get('details', {})
                    wr_random = details.get('random', {}).get('win_rate')
                    wr_negamax = details.get('negamax', {}).get('win_rate')
                    wr_random_s = f"{wr_random*100:.0f}%" if wr_random is not None else "N/A"
                    wr_negamax_s = f"{wr_negamax*100:.0f}%" if wr_negamax is not None else "N/A"
                    print(f"  Ep {start_ep + ep + 1:>6d}/{start_ep + num_episodes}  "
                          f"ε={epsilon:.3f}  "
                          f"Score={wr*100:.1f}%  "
                          f"WR(random)={wr_random_s}  "
                          f"WR(negamax)={wr_negamax_s}  "
                          f"t={elapsed:.0f}s  "
                          f"w={self.weights[1:4].round(4)}")

                if wr >= self.best_win_rate:
                    self.best_win_rate = wr
                    self.best_weights = self.weights.copy()

                self.history.append((start_ep + ep + 1, wr))

                if checkpoint_path:
                    save_checkpoint(self.weights, self.best_weights,
                                    self.episode, self.best_win_rate,
                                    self.history, checkpoint_path)

        if eval_count == 0:
            # No evaluation phase: keep final training weights as best checkpoint.
            self.best_weights = self.weights.copy()
        elif self.best_win_rate > 0:
            self.weights = self.best_weights.copy()

        if verbose:
            total_ep = start_ep + num_episodes
            if eval_count == 0:
                print(f"\nDone: {start_ep}→{total_ep} episodes, no evaluation phase.")
            else:
                print(f"\nDone: {start_ep}→{total_ep} episodes, "
                      f"best Score={self.best_win_rate*100:.1f}%")

        return self.weights

    def save(self, filepath: str = "agents/td_weights.py"):
        export_weights(self.best_weights, filepath)
        save_checkpoint(self.best_weights, self.best_weights,
                        self.episode, self.best_win_rate,
                        self.history, filepath)


# ── CLI ───────────────────────────────────────────────────────


def parse_eval_mix(eval_mix_str: str) -> List[Tuple[str, float]]:
    """
    Parse --eval-mix string like:
      "random:0.35,negamax:0.65"
    """
    parts = [p.strip() for p in (eval_mix_str or "").split(",") if p.strip()]
    if not parts:
        return [("random", 1.0)]
    result: List[Tuple[str, float]] = []
    for p in parts:
        if ":" in p:
            name, w = p.split(":", 1)
            try:
                weight = float(w.strip())
            except ValueError:
                continue
            result.append((name.strip(), weight))
        else:
            result.append((p, 1.0))
    return _normalize_eval_mix(result)

if __name__ == "__main__":
    import argparse

    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..'))

    parser = argparse.ArgumentParser(
        description="TD-Learning Evaluation Optimizer")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--lam", type=float, default=0.3)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument(
        "--eval-mix",
        type=str,
        default="random:0.35,negamax:0.65",
        help="Model-selection opponents and weights, e.g. random:0.35,negamax:0.65",
    )
    parser.add_argument("--epsilon-start", type=float, default=0.3)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--output", type=str, default="agents/td_weights.py")
    parser.add_argument(
        "--build-profile",
        choices=("none", "auto", "optuna", "td"),
        default="td",
        help="Build final submission after training (default: td).",
    )
    parser.add_argument(
        "--build-output",
        type=str,
        default="submission.py",
        help="Built single-file submission path.",
    )
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="从 --output 指定的文件恢复权重并继续训练")
    parser.add_argument("--load", type=str, default=None,
                        help="从指定权重文件加载并继续训练")
    args = parser.parse_args()

    print("=" * 60)
    print("  TD-Learning Evaluation Optimizer")
    print("=" * 60)
    print(f"  Episodes: {args.episodes}  α={args.alpha}  λ={args.lam}")
    print(f"  ε: {args.epsilon_start} → {args.epsilon_end}")
    print(f"  Eval: every {args.eval_interval} ep, {args.eval_games} games")
    print("=" * 60)

    if args.load:
        print(f"\nLoading weights from: {args.load}")
        trainer = TDTrainer.from_checkpoint(args.load)
    elif args.resume:
        print(f"\nResuming from: {args.output}")
        trainer = TDTrainer.from_checkpoint(args.output)
    else:
        trainer = TDTrainer()

    if trainer.episode > 0:
        print(f"  Resuming from episode {trainer.episode}, "
              f"best Score={trainer.best_win_rate*100:.1f}%")
    else:
        print(f"\nInitial weights: {trainer.weights.round(4).tolist()}\n")

    eval_mix = parse_eval_mix(args.eval_mix)
    print(f"  Eval mix: {', '.join(f'{n}:{w:.2f}' for n, w in eval_mix)}")

    trainer.train(
        num_episodes=args.episodes,
        alpha=args.alpha,
        lam=args.lam,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games if not args.no_eval else 0,
        verbose=True,
        checkpoint_path=args.output,
        eval_mix=eval_mix,
    )

    trainer.save(args.output)

    if args.build_profile != "none":
        root = Path(__file__).resolve().parents[1]
        build_output = Path(args.build_output)
        if not build_output.is_absolute():
            build_output = root / build_output
        td_weights_path = Path(args.output)
        if not td_weights_path.is_absolute():
            td_weights_path = root / td_weights_path
        cfg = build_submission(
            profile=args.build_profile,
            output_path=build_output,
            td_weights_path=td_weights_path,
        )
        print(f"[BUILD] profile={args.build_profile}")
        print(f"[BUILD] output={build_output}")
        print(f"[BUILD] source={cfg.source}")

    if not args.no_eval:
        print("\nFinal eval vs random (200 games)...")
        r = evaluate_against(trainer.best_weights, "random", 200)
        print(f"  {r['wins']}W/{r['losses']}L/{r['draws']}D  "
              f"WR={r['win_rate']*100:.1f}%")

        print("\nFinal eval vs negamax (200 games)...")
        r = evaluate_against(trainer.best_weights, "negamax", 200)
        print(f"  {r['wins']}W/{r['losses']}L/{r['draws']}D  "
              f"WR={r['win_rate']*100:.1f}%")
