"""ConnectX evaluation package.

Keep package import lightweight so build/config utilities can run without
optional runtime dependencies (e.g. kaggle_environments).
"""

__all__ = [
    "EloEngine",
    "TournamentRunner",
    "BenchmarkRunner",
    "Reporter",
    "TDTrainer",
]


def __getattr__(name):
    if name == "EloEngine":
        from .elo import EloEngine

        return EloEngine
    if name in ("TournamentRunner", "BenchmarkRunner"):
        from .tournament import BenchmarkRunner, TournamentRunner

        return {"TournamentRunner": TournamentRunner, "BenchmarkRunner": BenchmarkRunner}[name]
    if name == "Reporter":
        from .report import Reporter

        return Reporter
    if name == "TDTrainer":
        from .td_learn import TDTrainer

        return TDTrainer
    raise AttributeError(f"module 'evaluate' has no attribute {name!r}")

