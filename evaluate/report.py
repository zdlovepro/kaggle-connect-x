"""
结果报告生成器

支持:
  - 纯文本报告生成
  - JSON 导出 (可用于外部图表)
  - 对战历史记录
"""

import json
import os
from typing import Dict, List, Optional

from .elo import EloEngine
from .tournament import TournamentRunner


class Reporter:
    """结果报告生成器"""

    def __init__(self, output_dir: str = "evaluate/results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_text_report(
        self,
        elo: EloEngine,
        tournament: Optional[TournamentRunner] = None,
        benchmark_results: Optional[Dict] = None,
    ) -> str:
        """生成纯文本格式的综合报告"""
        sections = []

        # 标题
        sections.append("=" * 70)
        sections.append("  ConnectX Agent Evaluation Report")
        sections.append("=" * 70)

        # Elo 评分
        sections.append("\n## Elo Ratings")
        sections.append(elo.summary())

        # 锦标赛结果
        if tournament and tournament.results:
            sections.append("\n## Matchup Details")
            for matchup, result in tournament.results.items():
                sections.append(
                    f"  {matchup}: "
                    f"{result['wins_a']}W/{result['losses_a']}L/{result['draws_a']}D "
                    f"({result['win_rate_a']*100:.1f}%) "
                    f"[Elo Δ: {result['elo_delta_a']:+.1f}]"
                )

        # 性能基准
        if benchmark_results:
            sections.append("\n## Performance Benchmarks")
            for name, result in benchmark_results.items():
                sections.append(
                    f"  {name}: "
                    f"avg={result.get('time_ms_avg', 0):.1f}ms "
                    f"p99={result.get('time_ms_p99', 0):.1f}ms "
                    f"max={result.get('time_ms_max', 0):.1f}ms"
                )

        sections.append("\n" + "=" * 70)
        return "\n".join(sections)

    def export_json(
        self,
        elo: EloEngine,
        filename: str = "elo_ratings.json",
    ):
        """导出 Elo 评分为 JSON"""
        path = os.path.join(self.output_dir, filename)
        data = elo.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[Report] Elo ratings saved to {path}")

    def export_tournament_json(
        self,
        tournament: TournamentRunner,
        filename: str = "tournament_results.json",
    ):
        """导出锦标赛结果为 JSON"""
        path = os.path.join(self.output_dir, filename)
        # 将非序列化字段清理
        clean_results = {}
        for name, result in tournament.results.items():
            clean_results[name] = {k: v for k, v in result.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean_results, f, indent=2)
        print(f"[Report] Tournament results saved to {path}")

    def save_report(
        self,
        elo: EloEngine,
        tournament: Optional[TournamentRunner] = None,
        benchmark_results: Optional[Dict] = None,
        filename: str = "report.txt",
    ):
        """保存完整报告到文件"""
        report = self.generate_text_report(elo, tournament, benchmark_results)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[Report] Full report saved to {path}")
