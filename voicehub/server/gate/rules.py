"""Rule engine for gate pre-filtering.

Rules evaluated by priority (high first); first match decides.
Thread-safe; supports hot add/remove/update for self-evolution.
"""
from __future__ import annotations
import re
import threading
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """A single gate rule."""
    name: str
    pattern: str  # regex, matched against stripped transcript
    verdict: str  # "pass" | "block"
    priority: int = 0  # higher = evaluated first
    description: str = ""
    enabled: bool = True
    _compiled: re.Pattern = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._compiled = re.compile(self.pattern, re.IGNORECASE)

    def matches(self, text: str) -> bool:
        return bool(self._compiled.search(text))


# Initial rule set - evolves via offline optimization (see optimizer.py).
# Keep patterns compact; the LLM reviewer covers what rules miss.
INITIAL_RULES: list[Rule] = [
    # === BLOCK (high priority) ===
    Rule(
        name="self_talk_filler",
        pattern=r"^(?:唉|哎|啊|哦|噢|嗯+|呃|诶|嘿+|哈+|呵+|切|啧|嘘|哼|得嘞|得|好吧|算了|随便|无所谓|就这样|这样|那样|这个|那个|然后呢|接着|之后呢|对了|哎呀|我的天|天哪|好家伙|绝了|无语|服了|真行|可以啊|还行吧|一般般|凑合吧?|就那样|得得得|行了行了?|好了好了?)[。.!！~…\s]*$",
        verdict="block",
        priority=100,
        description="Self-talk fillers and sighs",
    ),
    Rule(
        name="short_ack",
        pattern=r"^(?:嗯|哦|噢|啊|对+|是+|行+|好+|好的?哦?|好滴|好嘞|好咧|好啊|可以|收到|明白|了解|知道|谢谢|感谢|多谢|不错|是的?嗯?|对的?啊?|对啊|可不是|那倒是|确实|当然|必须的?|嗯呢|嗯呐|嗯嗯+|哦哦+|噢噢+|哈哈+|呵呵+|嘿嘿+|噗|哇塞|牛|强|厉害)[。.!！~…\s]*$",
        verdict="block",
        priority=95,
        description="Short acknowledgements not worth a response",
    ),
    Rule(
        name="third_person_address",
        pattern=r"(?:^|[，,。！？\s])(?:你说|你觉得|你感觉|你想|你要|你干嘛|你干什么|你怎么|你咋|你为啥|你凭什么|你算什么|你到底|你究竟|你难道|你怎么能|你们说|你们觉得|他她说|妈|爸|老婆|老公|哥们|姐妹|师傅|同学|服务员)(?:说|觉得|看|想|干嘛|吃|来|去|听)?[，,？?\s]",
        verdict="block",
        priority=90,
        description="Addressing another person in the room",
    ),
    Rule(
        name="tv_broadcast",
        pattern=r"(?:观众朋友|各位观众|欢迎收看|欢迎收听|本期节目|广告之后|马上回来|精彩内容|不要走开|订阅频道|点赞关注|一键三连|下期再见|感谢观看)",
        verdict="block",
        priority=85,
        description="TV/radio/livestream speech patterns",
    ),
    # === PASS (lower priority) ===
    Rule(
        name="agent_alias",
        pattern=r"(?:小克|openclaw|克劳德?|赫尔墨斯|hermes|爱马仕|赫敏|小助|workbuddy|工作搭档|工作伙伴|小樱|樱桃|cherry)",
        verdict="pass",
        priority=105,  # above all block rules: naming an agent always wins
        description="Agent name mentioned - always respond",
    ),
    Rule(
        name="question_mark",
        pattern=r"[?？]$",
        verdict="pass",
        priority=45,
        description="Ends with question mark",
    ),
    Rule(
        name="question_words",
        pattern=r"(?:什么|怎么|怎样|怎么样|如何|为什么|为啥|哪个|哪里|哪儿|谁|多少|几[点号天个]|是不是|能不能|可不可以|有没有|要不要|行不行|好不好)",
        verdict="pass",
        priority=40,
        description="Contains question words",
    ),
    Rule(
        name="imperative_request",
        pattern=r"(?:帮我|给我|替我|麻烦你|请你|来一|打开|关闭|查一下|搜索|播放|暂停|停止|继续|重复|再说一遍|提醒我|记一下|设置|切换|调[高低]|增加|减少|开始|结束)",
        verdict="pass",
        priority=35,
        description="Imperative request patterns",
    ),
]


class RuleEngine:
    """Thread-safe rule engine with hot-reload."""

    def __init__(self, rules: Optional[list[Rule]] = None):
        self._rules: list[Rule] = list(rules or INITIAL_RULES)
        self._lock = threading.RLock()
        self._version = 0
        self._match_counts: dict[str, int] = {}
        self._sort_rules()

    def _sort_rules(self) -> None:
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    @property
    def version(self) -> int:
        return self._version

    @property
    def rules(self) -> list[Rule]:
        with self._lock:
            return list(self._rules)

    def evaluate(self, text: str) -> Optional[tuple[str, str]]:
        """Returns (verdict, rule_name) on first match, else None (no rule decided)."""
        with self._lock:
            for rule in self._rules:
                if rule.enabled and rule.matches(text):
                    self._match_counts[rule.name] = self._match_counts.get(rule.name, 0) + 1
                    return (rule.verdict, rule.name)
        return None

    def add_rule(self, rule: Rule) -> None:
        with self._lock:
            if any(r.name == rule.name for r in self._rules):
                return  # idempotent: evolution may re-propose the same rule
            self._rules.append(rule)
            self._sort_rules()
            self._version += 1

    def remove_rule(self, name: str) -> bool:
        with self._lock:
            before = len(self._rules)
            self._rules = [r for r in self._rules if r.name != name]
            if len(self._rules) < before:
                self._version += 1
                return True
        return False

    def update_rule(self, name: str, **kwargs) -> bool:
        with self._lock:
            for rule in self._rules:
                if rule.name == name:
                    for k, v in kwargs.items():
                        if k == "pattern":
                            rule.pattern = v
                            rule._compiled = re.compile(v, re.IGNORECASE)
                        elif hasattr(rule, k):
                            setattr(rule, k, v)
                    self._sort_rules()
                    self._version += 1
                    return True
        return False

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_rules": len(self._rules),
                "block_rules": sum(1 for r in self._rules if r.verdict == "block"),
                "pass_rules": sum(1 for r in self._rules if r.verdict == "pass"),
                "version": self._version,
                "match_counts": dict(self._match_counts),
            }
