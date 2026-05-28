"""
ConnectX 全面单元测试

覆盖:
  - submission.py 所有辅助函数
  - agents/zobrist.py (Zobrist 哈希 + 置换表)
  - agents/minimax_bitboard.py (bitboard 辅助函数 + evaluate)
  - agents/mcts_agent.py (辅助函数 + MCTSNode)
  - agents/base.py (BaseAgent 基类)
  - evaluate/elo.py (Elo 完整测试)
  - evaluate/tournament.py (BenchmarkRunner)
  - evaluate/report.py (JSON 导出)
  - evaluate/ablation.py (消融实验)
"""

import math
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

# ── 导入 submission.py 辅助函数 ──
from submission import (
    _has_won, _get_heights_from_list, _valid_cols, _drop_pos,
    _list_to_bitboards, _eval_window_count, _eval_odd_even_threats,
    _eval_position_heatmap, evaluate, _evaluate_learned,
    _order_moves as _sub_order_moves, _zobrist_init, _zobrist_update,
    _tt_store, _tt_probe, _tt_get_move,
    _killer_store, _is_killer,
    agent, _book_move,
)

# ── 导入 agents 模块 ──
from agents.base import BaseAgent
from agents.zobrist import (
    compute_hash,
    hash_update,
    hash_toggle_turn,
    TranspositionTable,
    ZOBRIST_TABLE,
    ZOBRIST_TURN,
    TT_FLAG_EXACT,
    TT_FLAG_LOWER,
    TT_FLAG_UPPER,
)
from connectx.bitboard import COLS, ROWS
from agents.minimax_bitboard import (
    _drop_piece_bb,
    _get_valid_cols,
    _is_draw,
    _count_open_threats,
    evaluate as _evaluate_bb,
    MinimaxBitboardAgent,
)
import agents.minimax_bitboard as _bb_mod
from agents.mcts_agent import (
    MCTSNode,
    MCTSAgent,
)

# ── 导入 evaluate 模块 ──
from evaluate.elo import EloEngine, EloRating
from evaluate.tournament import TournamentRunner, BenchmarkRunner, run_single_game
from evaluate.report import Reporter
from evaluate.ablation import run_ablation, ablation_suite


# =====================================================================
# submission.py 辅助函数测试
# =====================================================================

class TestGetValidActions(unittest.TestCase):
    """_valid_cols 测试 (基于 bitboard heights)"""

    def test_empty_board(self):
        board_list = [0] * 42
        heights = _get_heights_from_list(board_list)
        self.assertEqual(_valid_cols(heights), [0, 1, 2, 3, 4, 5, 6])

    def test_full_column_filtered(self):
        board_list = [0] * 42
        for r in range(6):
            board_list[r * 7 + 3] = 1
        heights = _get_heights_from_list(board_list)
        self.assertNotIn(3, _valid_cols(heights))

    def test_all_full(self):
        board_list = [1] * 42
        heights = _get_heights_from_list(board_list)
        self.assertEqual(_valid_cols(heights), [])


class TestDropPiece(unittest.TestCase):
    """_drop_pos 测试 (基于 bitboard)"""

    def test_drop_on_empty_column(self):
        b1, b2 = _list_to_bitboards([0] * 42)
        heights = _get_heights_from_list([0] * 42)
        pos = _drop_pos(heights, 3)
        self.assertEqual(int(pos), 5 * 7 + 3)  # row 5 (bottom), stride=7

    def test_drop_on_partial_column(self):
        board_list = [0] * 42
        board_list[5 * 7 + 3] = 1
        board_list[4 * 7 + 3] = 1
        heights = _get_heights_from_list(board_list)
        pos = _drop_pos(heights, 3)
        self.assertEqual(int(pos), 3 * 7 + 3)  # row 3, stride=7


class TestIsTerminal(unittest.TestCase):
    """_has_won 测试 (bitboard 赢棋检测)"""

    def test_no_win(self):
        board_list = [0] * 42
        board_list[5 * 7 + 3] = 1
        b1, _ = _list_to_bitboards(board_list)
        self.assertFalse(_has_won(b1))

    def test_horizontal_win(self):
        board_list = [0] * 42
        for c in range(4):
            board_list[5 * 7 + c] = 1
        b1, _ = _list_to_bitboards(board_list)
        self.assertTrue(_has_won(b1))

    def test_vertical_win(self):
        board_list = [0] * 42
        for r in range(2, 6):
            board_list[r * 7 + 3] = 1  # col 3, rows 2-5 → stride=7 vertical
        b1, _ = _list_to_bitboards(board_list)
        self.assertTrue(_has_won(b1))

    def test_diagonal_main_win(self):
        board_list = [0] * 42
        for i in range(4):
            board_list[(2 + i) * 7 + i] = 1  # (2,0)→(3,1)→(4,2)→(5,3)
        b1, _ = _list_to_bitboards(board_list)
        self.assertTrue(_has_won(b1))

    def test_diagonal_anti_win(self):
        board_list = [0] * 42
        for i in range(4):
            board_list[(2 + i) * 7 + (4 - i)] = 1  # (2,4)→(3,3)→(4,2)→(5,1)
        b1, _ = _list_to_bitboards(board_list)
        self.assertTrue(_has_won(b1))


class TestScoreWindow(unittest.TestCase):
    """_eval_window_count 测试 (bitboard 窗口计数)"""

    def test_empty_board(self):
        b1, b2 = _list_to_bitboards([0] * 42)
        score = _eval_window_count(b1, b2)
        self.assertEqual(score, 0)

    def test_single_piece_score(self):
        board_list = [0] * 42
        board_list[5 * 7 + 3] = 1  # center bottom
        b1, b2 = _list_to_bitboards(board_list)
        # score should include heatmap + window contribution
        score = evaluate(b1, b2, _get_heights_from_list(board_list))
        self.assertIsInstance(score, (int, float))

    def test_winning_position_positive(self):
        board_list = [0] * 42
        for c in range(4):
            board_list[5 * 7 + c] = 1
        b1, b2 = _list_to_bitboards(board_list)
        heights = _get_heights_from_list(board_list)
        self.assertTrue(_has_won(b1))
        score = evaluate(b1, b2, heights)
        self.assertTrue(score > 0 or score == 10000000)


class TestEvaluateBoard(unittest.TestCase):
    """evaluate() 测试 (bitboard 评估函数)"""

    def test_empty_board_score(self):
        board_list = [0] * 42
        b1, b2 = _list_to_bitboards(board_list)
        heights = _get_heights_from_list(board_list)
        score = evaluate(b1, b2, heights)
        self.assertIsInstance(score, (int, float))

    def test_winning_board_positive(self):
        board_list = [0] * 42
        for c in range(4):
            board_list[5 * 7 + c] = 1
        b1, b2 = _list_to_bitboards(board_list)
        heights = _get_heights_from_list(board_list)
        score = evaluate(b1, b2, heights)
        self.assertGreater(score, 0)

    def test_center_piece_bonus(self):
        board_list_c = [0] * 42
        board_list_c[5 * 7 + 3] = 1
        b1c, b2c = _list_to_bitboards(board_list_c)
        hc = _get_heights_from_list(board_list_c)
        score_center = evaluate(b1c, b2c, hc)

        board_list_e = [0] * 42
        board_list_e[5 * 7 + 0] = 1
        b1e, b2e = _list_to_bitboards(board_list_e)
        he = _get_heights_from_list(board_list_e)
        score_edge = evaluate(b1e, b2e, he)
        self.assertGreater(score_center, score_edge)


class TestBoardHash(unittest.TestCase):
    """Zobrist 哈希测试 (submission.py 内联实现)"""

    def test_same_position_same_hash(self):
        board_list = [0] * 42
        board_list[5 * 7 + 3] = 1
        b1, b2 = _list_to_bitboards(board_list)
        h1 = _zobrist_init(b1, b2)
        h2 = _zobrist_init(b1, b2)
        self.assertEqual(h1, h2)

    def test_different_position_different_hash(self):
        board_list1 = [0] * 42
        board_list2 = [0] * 42
        board_list2[5 * 7 + 3] = 1
        b1a, b2a = _list_to_bitboards(board_list1)
        b1b, b2b = _list_to_bitboards(board_list2)
        self.assertNotEqual(_zobrist_init(b1a, b2a), _zobrist_init(b1b, b2b))

    def test_update_changes_hash(self):
        board_list = [0] * 42
        board_list[5 * 7 + 3] = 1
        b1, b2 = _list_to_bitboards(board_list)
        h_before = _zobrist_init(b1, b2)
        pos = _drop_pos(_get_heights_from_list([0] * 42), 4)
        h_after = _zobrist_update(h_before, pos, 1)
        self.assertNotEqual(h_before, h_after)


class TestSubOrderMoves(unittest.TestCase):
    """submission._order_moves 测试"""

    def test_center_first(self):
        ordered = _sub_order_moves([0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(ordered[0], 3)

    def test_tt_best_first(self):
        ordered = _sub_order_moves([0, 1, 2, 3, 4, 5, 6], tt_move=0)
        self.assertEqual(ordered[0], 0)

    def test_tt_best_not_in_valid(self):
        ordered = _sub_order_moves([1, 2, 3], tt_move=6)
        self.assertEqual(ordered, [3, 2, 1])


# =====================================================================
# agents/base.py 测试
# =====================================================================

class TestBaseAgent(unittest.TestCase):
    """BaseAgent 基类测试"""

    def test_cannot_instantiate_abstract(self):
        with self.assertRaises(TypeError):
            BaseAgent()

    def test_concrete_subclass(self):
        class TestAgent(BaseAgent):
            def select_action(self, obs, cfg):
                return 3

        a = TestAgent(name="test")
        self.assertEqual(a.name, "test")
        self.assertEqual(a.select_action(None, None), 3)

    def test_stats_initial(self):
        class TestAgent(BaseAgent):
            def select_action(self, obs, cfg):
                return 3

        a = TestAgent()
        self.assertEqual(a.stats["nodes_visited"], 0)
        self.assertEqual(a.stats["moves_made"], 0)
        self.assertEqual(a.stats["total_time_ms"], 0.0)

    def test_reset_stats(self):
        class TestAgent(BaseAgent):
            def select_action(self, obs, cfg):
                return 3

        a = TestAgent()
        a.stats["moves_made"] = 10
        a.stats["total_time_ms"] = 100.0
        a.reset_stats()
        self.assertEqual(a.stats["moves_made"], 0)
        self.assertEqual(a.stats["total_time_ms"], 0.0)

    def test_get_avg_time_ms(self):
        class TestAgent(BaseAgent):
            def select_action(self, obs, cfg):
                return 3

        a = TestAgent()
        self.assertEqual(a.get_avg_time_ms(), 0.0)
        a.stats["moves_made"] = 5
        a.stats["total_time_ms"] = 250.0
        self.assertEqual(a.get_avg_time_ms(), 50.0)


# =====================================================================
# agents/zobrist.py 测试
# =====================================================================

class TestZobristHash(unittest.TestCase):
    """Zobrist 哈希函数测试"""

    def test_compute_hash_empty_board(self):
        b1 = np.uint64(0)
        b2 = np.uint64(0)
        h = compute_hash(b1, b2)
        self.assertIsInstance(h, np.uint64)

    def test_compute_hash_different_boards(self):
        b1 = np.uint64(1) << np.uint64(5 * 7 + 3)  # bottom center, ROW_STRIDE=7
        b2 = np.uint64(0)
        h1 = compute_hash(b1, b2)
        h2 = compute_hash(b2, b1)
        self.assertNotEqual(int(h1), int(h2))

    def test_compute_hash_with_turn(self):
        b1 = np.uint64(0)
        b2 = np.uint64(0)
        h_no_turn = compute_hash(b1, b2, turn_player=0)
        h_with_turn = compute_hash(b1, b2, turn_player=1)
        self.assertNotEqual(int(h_no_turn), int(h_with_turn))

    def test_hash_update_incremental(self):
        h = compute_hash(np.uint64(0), np.uint64(0))
        pos = 5 * 7 + 3
        h2 = hash_update(h, pos, 1)
        b1 = np.uint64(1) << np.uint64(pos)
        h_direct = compute_hash(b1, np.uint64(0))
        self.assertEqual(int(h2), int(h_direct))

    def test_hash_toggle_turn(self):
        h = compute_hash(np.uint64(0), np.uint64(0))
        h_toggled = hash_toggle_turn(h)
        h_toggled_back = hash_toggle_turn(h_toggled)
        self.assertEqual(int(h), int(h_toggled_back))

    def test_zobrist_table_exists(self):
        self.assertEqual(ZOBRIST_TABLE.shape, (42, 2))
        self.assertGreater(int(ZOBRIST_TURN), 0)


class TestTranspositionTable(unittest.TestCase):
    """TranspositionTable 测试"""

    def setUp(self):
        self.tt = TranspositionTable(size=1024)

    def test_store_and_probe_exact(self):
        key = np.uint64(12345)
        self.tt.store(key, 42.0, 5, TT_FLAG_EXACT, best_move=3)
        result = self.tt.probe(key, 3, -math.inf, math.inf)
        self.assertIsNotNone(result)
        val, move = result
        self.assertEqual(val, 42.0)
        self.assertEqual(move, 3)

    def test_probe_insufficient_depth(self):
        key = np.uint64(12345)
        self.tt.store(key, 42.0, 5, TT_FLAG_EXACT)
        result = self.tt.probe(key, 10, -math.inf, math.inf)
        self.assertIsNone(result)

    def test_probe_lower_bound_cutoff(self):
        key = np.uint64(12345)
        self.tt.store(key, 100.0, 5, TT_FLAG_LOWER)
        result = self.tt.probe(key, 3, -math.inf, 50.0)
        self.assertIsNotNone(result)
        val, _ = result
        self.assertEqual(val, 50.0)  # beta cutoff

    def test_probe_upper_bound_cutoff(self):
        key = np.uint64(12345)
        self.tt.store(key, 50.0, 5, TT_FLAG_UPPER)
        result = self.tt.probe(key, 3, 100.0, math.inf)
        self.assertIsNotNone(result)
        val, _ = result
        self.assertEqual(val, 100.0)  # alpha cutoff

    def test_probe_miss(self):
        key = np.uint64(99999)
        result = self.tt.probe(key, 5, -math.inf, math.inf)
        self.assertIsNone(result)

    def test_depth_preferred_replacement(self):
        key = np.uint64(12345)
        self.tt.store(key, 10.0, 3, TT_FLAG_EXACT)
        self.tt.store(key, 20.0, 8, TT_FLAG_EXACT)
        result = self.tt.probe(key, 5, -math.inf, math.inf)
        val, _ = result
        self.assertEqual(val, 20.0)

    def test_depth_not_replaced_by_shallower(self):
        key = np.uint64(12345)
        self.tt.store(key, 20.0, 8, TT_FLAG_EXACT)
        self.tt.store(key, 10.0, 3, TT_FLAG_EXACT)
        result = self.tt.probe(key, 5, -math.inf, math.inf)
        val, _ = result
        self.assertEqual(val, 20.0)  # deeper entry preserved

    def test_get_best_move(self):
        key = np.uint64(12345)
        self.tt.store(key, 42.0, 5, TT_FLAG_EXACT, best_move=6)
        self.assertEqual(self.tt.get_best_move(key), 6)

    def test_get_best_move_miss(self):
        self.assertEqual(self.tt.get_best_move(np.uint64(99999)), -1)

    def test_clear(self):
        key = np.uint64(12345)
        self.tt.store(key, 42.0, 5, TT_FLAG_EXACT)
        self.tt.clear()
        self.assertIsNone(self.tt.probe(key, 3, -math.inf, math.inf))


# =====================================================================
# agents/minimax_bitboard.py 辅助函数测试
# =====================================================================

class TestHasWon(unittest.TestCase):
    """_has_won 位运算获胜检测测试"""

    def test_empty_board(self):
        self.assertFalse(_has_won(np.uint64(0)))

    def test_horizontal_win(self):
        b = np.uint64(0)
        for c in range(4):
            pos = 5 * 7 + c
            b |= np.uint64(1) << np.uint64(pos)
        self.assertTrue(_has_won(b))

    def test_vertical_win(self):
        b = np.uint64(0)
        for r in range(2, 6):
            pos = r * 7 + 3
            b |= np.uint64(1) << np.uint64(pos)
        self.assertTrue(_has_won(b))

    def test_diagonal_win(self):
        b = np.uint64(0)
        # diagonal ↘: pos diff = ROWS+2 = 8, stride = ROWS+1 = 7
        for i in range(4):
            pos = (2 + i) * 7 + i  # (2,0),(3,1),(4,2),(5,3)
            b |= np.uint64(1) << np.uint64(pos)
        self.assertTrue(_has_won(b))

    def test_anti_diagonal_win(self):
        b = np.uint64(0)
        # anti-diagonal ↗ avoiding column 0 (masked by has_won)
        for i in range(4):
            pos = (5 - i) * 7 + (i + 1)  # (5,1),(4,2),(3,3),(2,4)
            b |= np.uint64(1) << np.uint64(pos)
        self.assertTrue(_has_won(b))

    def test_three_in_row_not_win(self):
        b = np.uint64(0)
        for c in range(3):
            pos = 5 * 7 + c
            b |= np.uint64(1) << np.uint64(pos)
        self.assertFalse(_has_won(b))


class TestListToBitboards(unittest.TestCase):
    """_list_to_bitboards 测试 (minimax_bitboard 版本, 返回 3 值)"""

    def test_empty_board(self):
        board = [0] * 42
        b1, b2, heights = _bb_mod._list_to_bitboards(board)
        self.assertEqual(int(b1), 0)
        self.assertEqual(int(b2), 0)
        self.assertTrue((heights == 0).all())

    def test_single_piece(self):
        board = [0] * 42
        board[5 * 7 + 3] = 1
        b1, b2, heights = _bb_mod._list_to_bitboards(board)
        self.assertGreater(int(b1), 0)
        self.assertEqual(int(b2), 0)
        self.assertEqual(heights[3], 1)

    def test_both_players(self):
        board = [0] * 42
        board[5 * 7 + 3] = 1
        board[4 * 7 + 3] = 2
        b1, b2, heights = _bb_mod._list_to_bitboards(board)
        self.assertEqual(heights[3], 2)
        self.assertGreater(int(b1), 0)
        self.assertGreater(int(b2), 0)

    def test_full_column(self):
        board = [0] * 42
        for r in range(6):
            board[r * 7 + 0] = 1
        b1, b2, heights = _bb_mod._list_to_bitboards(board)
        self.assertEqual(heights[0], 6)


class TestDropPieceBB(unittest.TestCase):
    """_drop_piece_bb 测试"""

    def test_empty_column(self):
        heights = np.zeros(7, dtype=np.uint8)
        pos = _drop_piece_bb(heights, 3)
        self.assertEqual(int(pos), 5 * 7 + 3)

    def test_partial_column(self):
        heights = np.zeros(7, dtype=np.uint8)
        heights[3] = 3
        pos = _drop_piece_bb(heights, 3)
        self.assertEqual(int(pos), 2 * 7 + 3)


class TestGetValidCols(unittest.TestCase):
    """_get_valid_cols (bitboard) 测试"""

    def test_all_valid(self):
        heights = np.zeros(7, dtype=np.uint8)
        self.assertEqual(_get_valid_cols(heights), [0, 1, 2, 3, 4, 5, 6])

    def test_full_column_excluded(self):
        heights = np.zeros(7, dtype=np.uint8)
        heights[3] = 6
        self.assertNotIn(3, _get_valid_cols(heights))


class TestIsDraw(unittest.TestCase):
    """_is_draw 测试"""

    def test_not_draw(self):
        heights = np.zeros(7, dtype=np.uint8)
        self.assertFalse(_is_draw(heights))

    def test_draw(self):
        heights = np.full(7, 6, dtype=np.uint8)
        self.assertTrue(_is_draw(heights))


class TestCountOpenThreats(unittest.TestCase):
    """_count_open_threats 测试"""

    def test_empty_board(self):
        self.assertEqual(_count_open_threats(np.uint64(0)), 0)

    def test_single_piece(self):
        b = np.uint64(1) << np.uint64(5 * 7 + 3)
        self.assertEqual(_count_open_threats(b), 0)

    def test_three_in_row(self):
        b = np.uint64(0)
        for c in range(3):
            pos = 5 * 7 + c
            b |= np.uint64(1) << np.uint64(pos)
        threats = _count_open_threats(b)
        self.assertGreater(threats, 0)


class TestMinimaxEvaluate(unittest.TestCase):
    """minimax_bitboard.evaluate 测试"""

    def test_empty_board(self):
        b_self = np.uint64(0)
        b_opp = np.uint64(0)
        heights = np.zeros(7, dtype=np.uint8)
        score = evaluate(b_self, b_opp, heights)
        self.assertEqual(score, 0.0)

    def test_self_win(self):
        b_self = np.uint64(0)
        for c in range(4):
            b_self |= np.uint64(1) << np.uint64(5 * 7 + c)
        b_opp = np.uint64(0)
        heights = np.zeros(7, dtype=np.uint8)
        heights[:4] = 1
        score = evaluate(b_self, b_opp, heights)
        self.assertGreater(score, 9000)

    def test_opp_win(self):
        b_self = np.uint64(0)
        b_opp = np.uint64(0)
        for c in range(4):
            b_opp |= np.uint64(1) << np.uint64(5 * 7 + c)
        heights = np.zeros(7, dtype=np.uint8)
        heights[:4] = 1
        score = evaluate(b_self, b_opp, heights)
        self.assertLess(score, -9000)


# =====================================================================
# agents/mcts_agent.py 辅助函数测试
# =====================================================================

class TestMCTSNode(unittest.TestCase):
    """MCTSNode 测试 (bitboard 状态)"""

    def test_node_creation(self):
        import numpy as np
        node = MCTSNode(b1=np.uint64(0), b2=np.uint64(0),
                        heights=np.zeros(7, dtype=np.int32), mark=1)
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.value, 0.0)

    def test_node_not_fully_expanded_initially(self):
        import numpy as np
        node = MCTSNode(b1=np.uint64(0), b2=np.uint64(0),
                        heights=np.zeros(7, dtype=np.int32), mark=1,
                        unvisited=[3, 4])
        self.assertFalse(node.is_fully_expanded())

    def test_node_fully_expanded_after_consuming_unvisited(self):
        import numpy as np
        node = MCTSNode(b1=np.uint64(0), b2=np.uint64(0),
                        heights=np.zeros(7, dtype=np.int32), mark=1,
                        unvisited=[3])
        node.unvisited.pop(0)
        self.assertTrue(node.is_fully_expanded())

    def test_node_parent_reference(self):
        import numpy as np
        parent = MCTSNode(b1=np.uint64(0), b2=np.uint64(0),
                          heights=np.zeros(7, dtype=np.int32), mark=1)
        child = MCTSNode(b1=np.uint64(0), b2=np.uint64(0),
                         heights=np.zeros(7, dtype=np.int32), mark=2,
                         parent=parent, action=3)
        self.assertIs(child.parent, parent)
        self.assertEqual(child.action, 3)

    def test_node_children_dict(self):
        import numpy as np
        parent = MCTSNode(b1=np.uint64(0), b2=np.uint64(0),
                          heights=np.zeros(7, dtype=np.int32), mark=1)
        child = MCTSNode(b1=np.uint64(0), b2=np.uint64(0),
                         heights=np.zeros(7, dtype=np.int32), mark=2,
                         parent=parent, action=3)
        parent.children[3] = child
        self.assertEqual(len(parent.children), 1)
        self.assertIn(3, parent.children)


# =====================================================================
# evaluate/elo.py 扩展测试
# =====================================================================

class TestEloEngineExtended(unittest.TestCase):
    """Elo 评分引擎扩展测试"""

    def setUp(self):
        self.engine = EloEngine()

    def test_register_new(self):
        r = self.engine.register("A")
        self.assertEqual(r.name, "A")
        self.assertEqual(r.mu, 1000.0)
        self.assertEqual(r.sigma, 350.0)
        self.assertEqual(r.games, 0)

    def test_register_custom_mu(self):
        r = self.engine.register("B", initial_mu=1200)
        self.assertEqual(r.mu, 1200)

    def test_register_existing(self):
        self.engine.register("A", initial_mu=1000)
        r = self.engine.register("A", initial_mu=1500)
        self.assertEqual(r.mu, 1000)  # does not overwrite

    def test_get_rating_missing(self):
        self.assertIsNone(self.engine.get_rating("nonexistent"))

    def test_update_batch(self):
        self.engine.register("A")
        self.engine.register("B")
        self.engine.update_batch("A", "B", [1.0, 1.0, 0.0, 0.5])
        r = self.engine.get_rating("A")
        self.assertEqual(r.games, 4)

    def test_ranking_order(self):
        self.engine.register("A", initial_mu=1000)
        self.engine.register("B", initial_mu=1200)
        self.engine.register("C", initial_mu=1100)
        ranked = self.engine.ranking()
        self.assertEqual(ranked[0].name, "B")
        self.assertEqual(ranked[1].name, "C")
        self.assertEqual(ranked[2].name, "A")

    def test_to_dict(self):
        self.engine.register("A")
        self.engine.register("B")
        d = self.engine.to_dict()
        self.assertIn("A", d)
        self.assertIn("mu", d["A"])

    def test_save_snapshot(self):
        self.engine.register("A")
        self.engine.save_snapshot("A")
        r = self.engine.get_rating("A")
        self.assertEqual(len(r.history), 1)

    def test_save_snapshot_missing(self):
        self.engine.save_snapshot("nonexistent")  # should not raise

    def test_dynamic_k_factor(self):
        self.engine.register("A")  # sigma = 350
        self.engine.register("B")
        mu_before = self.engine.get_rating("A").mu
        self.engine.update("A", "B", 1.0)
        mu_after = self.engine.get_rating("A").mu
        # High sigma → k_factor=32 fully applied, delta = 32*(1-0.5) = 16
        delta = mu_after - mu_before
        self.assertGreater(delta, 15)

    def test_sigma_min_bound(self):
        self.engine.register("A")
        r = self.engine.get_rating("A")
        for _ in range(1000):
            r.sigma = max(EloEngine.MIN_SIGMA, r.sigma * 0.99)
        self.assertGreaterEqual(r.sigma, EloEngine.MIN_SIGMA)

    def test_win_rate_none_for_no_games(self):
        self.engine.register("A")
        self.assertIsNone(self.engine.win_rate("A"))


# =====================================================================
# evaluate/tournament.py 测试
# =====================================================================

class TestBenchmarkRunner(unittest.TestCase):
    """BenchmarkRunner 测试"""

    def test_benchmark_basic(self):
        runner = BenchmarkRunner()

        def simple_agent(board):
            return 3

        boards = [[0] * 42, [0] * 42]
        result = runner.benchmark_agent(simple_agent, boards, repetitions=3)
        self.assertIn("time_ms_avg", result)
        self.assertIn("samples", result)
        self.assertEqual(result["samples"], 6)

    def test_benchmark_compare(self):
        runner = BenchmarkRunner()
        agent_results = {
            "AgentA": {"time_ms_avg": 50.0, "time_ms_median": 48.0,
                       "time_ms_min": 10.0, "time_ms_max": 200.0, "time_ms_p99": 180.0},
            "AgentB": {"time_ms_avg": 25.0, "time_ms_median": 24.0,
                       "time_ms_min": 5.0, "time_ms_max": 100.0, "time_ms_p99": 90.0},
        }
        report = runner.compare(agent_results)
        self.assertIn("AgentA", report)
        self.assertIn("AgentB", report)


# =====================================================================
# evaluate/report.py 测试
# =====================================================================

class TestReporterExtended(unittest.TestCase):
    """Reporter 扩展测试"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.reporter = Reporter(output_dir=self.tmpdir)

    def test_export_json_creates_file(self):
        engine = EloEngine()
        engine.register("A")
        engine.register("B")
        engine.update("A", "B", 1.0)
        self.reporter.export_json(engine, filename="test_elo.json")
        path = os.path.join(self.tmpdir, "test_elo.json")
        self.assertTrue(os.path.exists(path))

    def test_export_tournament_json(self):
        runner = TournamentRunner()
        runner.run_matchup("submission.py", "random", games=2, verbose=False)
        self.reporter.export_tournament_json(runner, filename="test_tournament.json")
        path = os.path.join(self.tmpdir, "test_tournament.json")
        self.assertTrue(os.path.exists(path))

    def test_save_report(self):
        engine = EloEngine()
        engine.register("A")
        engine.register("B")
        self.reporter.save_report(engine, filename="test_report.txt")
        path = os.path.join(self.tmpdir, "test_report.txt")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Elo Ratings", content)

    def test_generate_text_report_with_tournament(self):
        engine = EloEngine()
        engine.register("A")
        engine.register("B")
        runner = TournamentRunner(elo=engine)
        runner.run_matchup("submission.py", "random", games=2, verbose=False)
        text = self.reporter.generate_text_report(engine, tournament=runner)
        self.assertIn("Elo Ratings", text)
        self.assertIn("Matchup Details", text)

    def test_generate_text_report_with_benchmark(self):
        engine = EloEngine()
        engine.register("A")
        benchmark = {"AgentA": {"time_ms_avg": 50.0, "time_ms_p99": 180.0, "time_ms_max": 200.0}}
        text = self.reporter.generate_text_report(engine, benchmark_results=benchmark)
        self.assertIn("Performance Benchmarks", text)


# =====================================================================
# evaluate/ablation.py 测试
# =====================================================================

class TestAblation(unittest.TestCase):
    """消融实验测试"""

    def test_run_ablation_basic(self):
        result = run_ablation(
            "submission.py", "random",
            games=4,
            label="test_ablation",
            verbose=False,
        )
        self.assertIn("win_rate_full", result)
        self.assertEqual(result["games"], 4)

    def test_ablation_suite_returns_results(self):
        results = ablation_suite(base_agent="submission.py", games=4, verbose=False)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("win_rate_full", r)
            self.assertIn("games", r)


# =====================================================================
# EloRating dataclass 测试
# =====================================================================

class TestEloRating(unittest.TestCase):
    """EloRating dataclass 测试"""

    def test_default_values(self):
        r = EloRating(name="test")
        self.assertEqual(r.name, "test")
        self.assertEqual(r.mu, 1000.0)
        self.assertEqual(r.sigma, 350.0)
        self.assertEqual(r.games, 0)
        self.assertEqual(r.wins, 0)
        self.assertEqual(r.losses, 0)
        self.assertEqual(r.draws, 0)
        self.assertEqual(r.history, [])

    def test_field_independence(self):
        r1 = EloRating(name="A")
        r2 = EloRating(name="B")
        r1.mu = 1200
        self.assertEqual(r2.mu, 1000.0)


# =====================================================================
# MinimaxBitboardAgent 扩展测试
# =====================================================================

class TestMinimaxBitboardAgentExtended(unittest.TestCase):
    """Bitboard Agent 扩展测试"""

    def setUp(self):
        self.columns = 7
        self.rows = 6
        self.cfg = Mock(columns=self.columns, rows=self.rows, inarow=4)

    def _make_obs(self, board, mark=1):
        return Mock(board=board, mark=mark, remainingOverageTime=120)

    def test_opening_book_first_move(self):
        agent_bb = MinimaxBitboardAgent(time_budget_ms=500)
        obs = self._make_obs([0] * 42)
        action = agent_bb.select_action(obs, self.cfg)
        self.assertEqual(action, 3)

    def test_opening_book_second_move(self):
        agent_bb = MinimaxBitboardAgent(time_budget_ms=500)
        board = [0] * 42
        board[5 * 7 + 3] = 1
        obs = self._make_obs(board, mark=2)
        action = agent_bb.select_action(obs, self.cfg)
        self.assertEqual(action, 3)

    def test_stats_accumulate(self):
        agent_bb = MinimaxBitboardAgent(time_budget_ms=500)
        obs = self._make_obs([0] * 42)
        agent_bb.select_action(obs, self.cfg)
        self.assertGreater(agent_bb.stats["moves_made"], 0)

    def test_instant_win_over_search(self):
        agent_bb = MinimaxBitboardAgent(time_budget_ms=500)
        board = [0] * 42
        base = 5 * 7
        for c in [0, 1, 2]:
            board[base + c] = 1
        obs = self._make_obs(board, mark=1)
        action = agent_bb.select_action(obs, self.cfg)
        self.assertEqual(action, 3)


# =====================================================================
# MCTSAgent 扩展测试
# =====================================================================

class TestMCTSAgentExtended(unittest.TestCase):
    """MCTS Agent 扩展测试"""

    def setUp(self):
        self.columns = 7
        self.rows = 6
        self.cfg = Mock(columns=self.columns, rows=self.rows, inarow=4)

    def _make_obs(self, board, mark=1):
        return Mock(board=board, mark=mark, remainingOverageTime=120)

    def test_instant_win(self):
        mcts = MCTSAgent(time_budget_ms=500)
        board = [0] * 42
        base = 5 * 7
        for c in [0, 1, 2]:
            board[base + c] = 1
        obs = self._make_obs(board, mark=1)
        action = mcts.select_action(obs, self.cfg)
        self.assertEqual(action, 3)

    def test_instant_block(self):
        mcts = MCTSAgent(time_budget_ms=500)
        board = [0] * 42
        base = 5 * 7
        for c in [1, 2, 3]:
            board[base + c] = 2
        obs = self._make_obs(board, mark=1)
        action = mcts.select_action(obs, self.cfg)
        self.assertIn(action, [0, 4])

    def test_stats_accumulate(self):
        mcts = MCTSAgent(time_budget_ms=500)
        obs = self._make_obs([0] * 42)
        mcts.select_action(obs, self.cfg)
        self.assertGreater(mcts.stats["moves_made"], 0)

    def test_empty_valid_actions(self):
        mcts = MCTSAgent(time_budget_ms=500)
        board = [1] * 42
        obs = self._make_obs(board)
        action = mcts.select_action(obs, self.cfg)
        self.assertEqual(action, 0)


# =====================================================================
# 入口
# =====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
