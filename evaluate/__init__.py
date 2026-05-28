"""
ConnectX 效果评估框架

提供:
  - Elo 评分引擎
  - 锦标赛运行器
  - 性能基准测试
  - 消融实验
  - 超参优化 (Optuna)
  - 结果报告生成
"""

from .elo import EloEngine
from .tournament import TournamentRunner, BenchmarkRunner
from .report import Reporter
from .td_learn import TDTrainer
