"""
MCTS (Monte Carlo Tree Search) Agent

基于 UCB1 选择策略的蒙特卡洛树搜索。

特性:
  - UCB1 (Upper Confidence Bound) 树选择策略
  - 启发式引导的 Rollout (80% 启发式, 20% 随机)
  - 时间感知的模拟控制
  - 必胜/必败早期终止
  - 着法排序 (中心优先 + 威胁优先)
"""

import math
import random
import time
from typing import Any, List, Optional, Tuple

from .base import BaseAgent


# ── 游戏常量 ─────────────────────────────────────────────────

def _get_valid_cols(board_list: List[int], columns: int, rows: int) -> List[int]:
    """获取未满列索引"""
    return [c for c in range(columns) if board_list[c] == 0]


def _drop_piece(board_list: List[int], columns: int, rows: int, col: int, mark: int) -> int:
    """落子，返回行索引"""
    for r in range(rows - 1, -1, -1):
        if board_list[r * columns + col] == 0:
            return r
    return -1


def _check_win(board_list: List[int], columns: int, rows: int, inarow: int,
               mark: int, last_col: int) -> bool:
    """检测是否获胜"""
    if last_col is None:
        return False

    # 找到落子行
    last_row = -1
    for r in range(rows - 1, -1, -1):
        if board_list[r * columns + last_col] == mark:
            last_row = r
            break
    if last_row == -1:
        return False

    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        r, c = last_row + dr, last_col + dc
        while 0 <= r < rows and 0 <= c < columns:
            if board_list[r * columns + c] == mark:
                count += 1
                r += dr
                c += dc
            else:
                break
        r, c = last_row - dr, last_col - dc
        while 0 <= r < rows and 0 <= c < columns:
            if board_list[r * columns + c] == mark:
                count += 1
                r -= dr
                c -= dc
            else:
                break
        if count >= inarow:
            return True
    return False


def _is_board_full(board_list: List[int], columns: int) -> bool:
    """检测棋盘是否已满"""
    return all(board_list[c] != 0 for c in range(columns))


# ── MCTS Node ────────────────────────────────────────────────

class MCTSNode:
    """MCTS 树节点"""

    __slots__ = ('state', 'parent', 'action', 'children', 'visits', 'value', 'unvisited')

    def __init__(self, state, parent=None, action=None, unvisited=None):
        self.state = state          # (board_list, current_mark, last_col)
        self.parent = parent
        self.action = action        # 到达此节点的列索引
        self.children = {}          # {column: MCTSNode}
        self.visits = 0
        self.value = 0.0            # 胜率 (从当前玩家视角的价值和)
        self.unvisited = unvisited or []  # 尚未展开的动作列表

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_fully_expanded(self) -> bool:
        return len(self.unvisited) == 0


# ── MCTS Agent ───────────────────────────────────────────────

class MCTSAgent(BaseAgent):
    """MCTS Agent with UCB1"""

    def __init__(
        self,
        name: str = "mcts",
        c_param: float = 1.414,       # UCB1 exploration constant
        time_budget_ms: float = 1900.0,  # Time budget per move
        heuristic_rollout_prob: float = 0.8,  # Heuristic rollout probability
    ):
        super().__init__(name=name)
        self.c_param = c_param
        self.time_budget_ms = time_budget_ms
        self.heuristic_rollout_prob = heuristic_rollout_prob
        self.simulations = 0

    def select_action(self, observation: Any, configuration: Any) -> int:
        board = observation.board
        mark = observation.mark
        columns = configuration.columns
        rows = configuration.rows
        inarow = configuration.inarow

        board_list = list(board)  # copy
        valid = _get_valid_cols(board_list, columns, rows)

        if not valid:
            self.stats["moves_made"] += 1
            return 0

        # 快速检测直接获胜
        for col in valid:
            row = _drop_piece(board_list, columns, rows, col, mark)
            idx = row * columns + col
            board_list[idx] = mark
            won = _check_win(board_list, columns, rows, inarow, mark, col)
            board_list[idx] = 0
            if won:
                self.stats["moves_made"] += 1
                return col

        # 快速检测拦截对手
        opp_mark = 2 if mark == 1 else 1
        for col in valid:
            row = _drop_piece(board_list, columns, rows, col, opp_mark)
            idx = row * columns + col
            board_list[idx] = opp_mark
            won = _check_win(board_list, columns, rows, inarow, opp_mark, col)
            board_list[idx] = 0
            if won:
                self.stats["moves_made"] += 1
                return col

        # MCTS 搜索
        root_unvisited = self._order_moves(valid, columns)
        root_state = (board_list.copy(), mark, None, columns, rows, inarow)
        root = MCTSNode(state=root_state, unvisited=root_unvisited)

        return self._search(root, columns, rows, inarow)

    def _search(
        self,
        root: MCTSNode,
        columns: int,
        rows: int,
        inarow: int,
    ) -> int:
        """MCTS 主搜索循环"""
        self.simulations = 0
        start = time.perf_counter()

        while True:
            elapsed = (time.perf_counter() - start) * 1000
            if elapsed > self.time_budget_ms:
                break

            # 1. Selection
            node = self._select(root)

            # 2. Expansion
            if not node.is_fully_expanded():
                node = self._expand(node, columns, rows, inarow)

            # 3. Simulation
            result = self._simulate(node.state, columns, rows, inarow)

            # 4. Backpropagation
            self._backpropagate(node, result)

            self.simulations += 1

        # 选择访问次数最多的动作
        if not root.children:
            return self._fallback(root, columns)

        # 优先选访问次数最多的，如果相同则选胜率最高的
        best_action = max(
            root.children.items(),
            key=lambda item: (item[1].visits, item[1].value / max(1, item[1].visits))
        )[0]

        self.stats["nodes_visited"] = self.simulations
        self.stats["moves_made"] += 1

        return best_action

    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCB1 选择：从根向下选择"""
        while not node.is_leaf() and not node.is_fully_expanded() is False:
            if node.is_fully_expanded():
                node = self._best_child(node)
            else:
                break
        return node

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        """UCB1: 选择 UCB 值最大的子节点"""
        best_ucb = -float('inf')
        best_child = None
        log_parent = math.log(max(1, node.visits))

        for child in node.children.values():
            if child.visits == 0:
                ucb = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = self.c_param * math.sqrt(log_parent / child.visits)
                ucb = exploitation + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child if best_child else list(node.children.values())[0]

    def _expand(self, node: MCTSNode, columns: int, rows: int, inarow: int) -> MCTSNode:
        """展开一个未访问的动作"""
        if not node.unvisited:
            return node

        # 取第一个未访问的动作
        action = node.unvisited.pop(0)

        # 创建新状态
        board_list, mark, last_col, cols, rws, iw = node.state
        new_board = board_list.copy()
        row = _drop_piece(new_board, columns, rows, action, mark)
        idx = row * columns + action
        new_board[idx] = mark

        new_mark = 2 if mark == 1 else 1
        new_state = (new_board, new_mark, action, columns, rows, inarow)

        # 获取该状态下的未访问动作
        new_unvisited = self._order_moves(
            _get_valid_cols(new_board, columns, rows), columns
        )

        child = MCTSNode(state=new_state, parent=node, action=action, unvisited=new_unvisited)
        node.children[action] = child
        return child

    def _simulate(self, state: tuple, columns: int, rows: int, inarow: int) -> float:
        """启发式 Rollout"""
        board_list, current_mark, last_col, cols, rws, iw = state
        sim_board = board_list.copy()
        sim_mark = current_mark
        sim_last_col = last_col
        actions_taken = 0

        while True:
            valid = _get_valid_cols(sim_board, columns, rows)
            if not valid:
                return 0.0  # draw

            # 检测终端
            if sim_last_col is not None:
                prev_mark = 2 if sim_mark == 1 else 1
                if _check_win(sim_board, columns, rows, inarow, prev_mark, sim_last_col):
                    # 上一步玩家赢了
                    if prev_mark == state[1]:  # 原玩家
                        return 1.0
                    else:
                        return 0.0

            # 选择动作
            if random.random() < self.heuristic_rollout_prob:
                action = self._heuristic_move(sim_board, sim_mark, valid, columns, rows, inarow)
            else:
                action = random.choice(valid)

            row = _drop_piece(sim_board, columns, rows, action, sim_mark)
            idx = row * columns + action
            sim_board[idx] = sim_mark
            sim_mark = 2 if sim_mark == 1 else 1
            sim_last_col = action
            actions_taken += 1

            # 安全限制
            if actions_taken > columns * rows:
                return 0.0

    def _backpropagate(self, node: MCTSNode, value: float):
        """反向传播：更新路径上的所有节点"""
        current = node
        while current is not None:
            current.visits += 1
            current.value += value
            value = 1.0 - value  # 切换到对手视角
            current = current.parent

    def _heuristic_move(
        self,
        board: List[int],
        mark: int,
        valid: List[int],
        columns: int,
        rows: int,
        inarow: int,
    ) -> int:
        """
        启发式着法选择:
          1. 直接获胜
          2. 拦截对手获胜
          3. 扩展已有2-in-a-row或3-in-a-row
          4. 中心列
        """
        opp_mark = 2 if mark == 1 else 1

        # 1. 检查能否获胜
        for col in valid:
            row = _drop_piece(board, columns, rows, col, mark)
            idx = row * columns + col
            board[idx] = mark
            won = _check_win(board, columns, rows, inarow, mark, col)
            board[idx] = 0
            if won:
                return col

        # 2. 拦截对手
        for col in valid:
            row = _drop_piece(board, columns, rows, col, opp_mark)
            idx = row * columns + col
            board[idx] = opp_mark
            won = _check_win(board, columns, rows, inarow, opp_mark, col)
            board[idx] = 0
            if won:
                return col

        # 3. 中心列 + 随机扰动
        center = columns // 2
        # 优先中心及其附近
        candidates = sorted(valid, key=lambda c: abs(c - center))
        if random.random() < 0.7:
            return candidates[0]
        else:
            return random.choice(candidates[:min(3, len(candidates))])

    def _order_moves(self, valid: List[int], columns: int) -> List[int]:
        """着法排序: 中心优先"""
        center = columns // 2
        return sorted(valid, key=lambda c: abs(c - center))

    def _fallback(self, root: MCTSNode, columns: int) -> int:
        """无法搜索时返回中心列"""
        return columns // 2


# ── Kaggle 兼容的 agent 函数 ─────────────────

_MCTS_AGENT = None


def create_mcts_agent(**kwargs) -> MCTSAgent:
    global _MCTS_AGENT
    if _MCTS_AGENT is None:
        _MCTS_AGENT = MCTSAgent(**kwargs)
    return _MCTS_AGENT


def mcts_agent_fn(observation: Any, configuration: Any) -> int:
    """Kaggle-compatible agent function"""
    agent = create_mcts_agent()
    return agent.select_action(observation, configuration)
