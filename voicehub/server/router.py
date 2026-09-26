"""Name-based agent routing + collaboration intent detection.

Routing: STT transcript -> match agent alias -> route to that agent.
If no alias matched, fall back to the currently active agent (continuous dialogue).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

COLLAB_WORDS = [
    "一起", "和", "跟", "加上", "然后让", "再让", "还要", "以及",
    "检查", "评审", "对比", "比较", "讨论", "圆桌", "一起聊聊", "协作",
]

# Collaboration mode triggers. Review wins over handoff wins over roundtable.
REVIEW_WORDS = ["评审", "审阅", "审查", "批评", "点评", "评一下", "评一评", "复核", "检查"]
HANDOFF_WORDS = ["然后让", "再让", "交给", "转给", "接力", "接着让", "之后让", "随后让"]

# Union of all words that can open a collaboration (requires >= 2 named agents).
# Without merging the mode words here, most review/handoff triggers
# (审阅/交给/评一下/...) would never reach the collaboration branch.
_COLLAB_TRIGGER = COLLAB_WORDS + HANDOFF_WORDS + REVIEW_WORDS


@dataclass
class RouteResult:
    agent: str
    text: str
    switched: bool
    collab: bool = False
    collab_agents: list[str] = field(default_factory=list)
    collab_mode: str = "roundtable"


class Router:
    def __init__(self, config: dict):
        self.agents = config["agents"]
        self.default = config.get("default_agent") or next(iter(self.agents))
        self.index: list[tuple[str, str]] = []
        for key, a in self.agents.items():
            if a.get("disabled"):
                continue
            for alias in a.get("aliases", []):
                self.index.append((self._norm(alias), key))
            disp = a.get("display")
            if disp:
                self.index.append((self._norm(disp), key))
        self.active = self.default

    @staticmethod
    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    def _alias_hits(self, text: str) -> list[str]:
        norm = self._norm(text)
        hits: list[str] = []
        for alias, key in self.index:
            if alias and alias in norm and key not in hits:
                hits.append(key)
        return hits

    def _ordered_hits(self, text: str) -> list[str]:
        """Agent keys present in text, ordered by first mention (utterance order).

        Collaboration direction (who acts on whom) follows the spoken order, not
        the config/index order, so review/handoff run the right way around.
        """
        norm = self._norm(text)
        best_pos: dict[str, int] = {}
        for alias, key in self.index:
            pos = norm.find(alias)
            if pos >= 0 and (key not in best_pos or pos < best_pos[key]):
                best_pos[key] = pos
        return [k for k, _ in sorted(best_pos.items(), key=lambda kv: kv[1])]

    def is_collab(self, text: str, hits: list[str] | None = None) -> bool:
        if hits is None:
            hits = self._alias_hits(text)
        has_word = any(w in text for w in _COLLAB_TRIGGER)
        return len(hits) >= 2 and has_word

    def _collab_mode(self, text: str) -> str:
        if any(w in text for w in REVIEW_WORDS):
            return "review"
        if any(w in text for w in HANDOFF_WORDS):
            return "handoff"
        return "roundtable"

    def route(self, text: str, active: Optional[str] = None) -> RouteResult:
        # `active` is the per-connection active agent (caller-owned state).
        # When omitted, fall back to the shared default and update it (legacy /
        # single-stream callers). Passing `active` keeps multi-connection state
        # isolated: the shared `self.active` is then never mutated by those calls.
        if active is None:
            active = self.active
        hits = self._alias_hits(text)
        collab = self.is_collab(text, hits)
        mode = self._collab_mode(text) if collab else "roundtable"
        ordered = self._ordered_hits(text) if collab else hits
        key = self._match_single(text)
        if key:
            switched = key != active
            if active is None:
                self.active = key
            return RouteResult(
                agent=key, text=self._strip(text, key),
                switched=switched, collab=collab, collab_agents=ordered,
                collab_mode=mode,
            )
        # no single alias -> stay on active agent; still flag collab if 2+ named
        return RouteResult(
            agent=active, text=text, switched=False,
            collab=collab, collab_agents=ordered, collab_mode=mode,
        )

    def _match_single(self, text: str) -> Optional[str]:
        norm = self._norm(text)
        for alias, key in self.index:
            if alias and alias in norm:
                return key
        return None

    def _strip(self, text: str, key: str) -> str:
        a = self.agents[key]
        names = a.get("aliases", []) + [a.get("display", "")]
        out = text
        for n in names:
            out = re.sub(re.escape(n), "", out, flags=re.IGNORECASE)
        return out.strip() or text
