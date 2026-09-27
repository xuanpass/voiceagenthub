"""Gate module: rule pre-filter + LLM review + rule self-evolution.

Pipeline position: STT -> GateProcessor -> Router
Rules block ~70% obvious non-addressed speech; LLM reviews the ambiguous rest.
"""
from .processor import GateProcessor
from .rules import RuleEngine, Rule
from .logger import GateLogger, GateRecord
from .reviewer import LLMReviewer, BaseReviewer
from .optimizer import RuleOptimizer

__all__ = [
    "GateProcessor",
    "RuleEngine",
    "Rule",
    "GateLogger",
    "GateRecord",
    "LLMReviewer",
    "BaseReviewer",
    "RuleOptimizer",
]
