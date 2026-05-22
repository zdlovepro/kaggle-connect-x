"""
Agent 抽象基类

所有 Agent 必须实现 select_action 方法。
统一接口方便评估框架调用。
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """ConnectX Agent 基类"""

    def __init__(self, name: str = "base"):
        self.name = name
        self.stats = {"nodes_visited": 0, "moves_made": 0, "total_time_ms": 0.0}

    @abstractmethod
    def select_action(self, observation: Any, configuration: Any) -> int:
        """
        选择落子列索引。

        Args:
            observation: 环境观察对象 (含 board, mark 等)
            configuration: 配置对象 (含 columns, rows, inarow)

        Returns:
            int: 列索引 [0, configuration.columns)
        """
        pass

    def reset_stats(self):
        """重置统计计数器"""
        self.stats = {"nodes_visited": 0, "moves_made": 0, "total_time_ms": 0.0}

    def get_avg_time_ms(self) -> float:
        """平均每步耗时 (毫秒)"""
        if self.stats["moves_made"] == 0:
            return 0.0
        return self.stats["total_time_ms"] / self.stats["moves_made"]
