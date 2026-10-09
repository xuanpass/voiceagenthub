import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.config import load_config
from server.router import Router


@pytest.fixture
def router():
    cfg = load_config()
    return Router(cfg)


def test_config_loads(router):
    assert set(router.agents.keys()) == {"openclaw", "hermes", "workbuddy", "cherrystudio"}


def test_route_by_alias(router):
    # agents.yaml 的 default_agent 现为 hermes；若不传 active，route() 默认用
    # default(=hermes) 作为当前 agent，此时 key==active，switched 恒为 False。
    # 显式传 active="openclaw" 模拟"从别的智能体唤醒赫尔墨斯"，才能验证别名
    # 路由确实产生了切换（switched=True）。
    r = router.route("赫尔墨斯，帮我写个总结", active="openclaw")
    assert r.agent == "hermes"
    assert r.switched is True
    assert "赫尔墨斯" not in r.text
    # 同 agent 场景（active 已是 hermes）不应算切换，switched 必须为 False。
    r2 = router.route("赫尔墨斯，再帮我写一个", active="hermes")
    assert r2.agent == "hermes"
    assert r2.switched is False


def test_route_fallback_active(router):
    router.active = "openclaw"
    r = router.route("今天天气怎么样")
    assert r.agent == "openclaw"  # no alias -> active


def test_collab_detection(router):
    text = "让小克和赫尔墨斯一起讨论这个问题"
    assert router.is_collab(text) is True
    hits = router._alias_hits(text)
    assert "openclaw" in hits and "hermes" in hits


def test_no_false_collab(router):
    assert router.is_collab("小克帮我查下资料") is False


def test_collab_via_review_word(router):
    # 审阅 is a review word but NOT in COLLAB_WORDS; must still open collab.
    r = router.route("让小樱审阅赫尔墨斯")
    assert r.collab is True
    assert r.collab_mode == "review"


def test_collab_via_handoff_word(router):
    # 交给 is a handoff word but NOT in COLLAB_WORDS; must still open collab.
    r = router.route("小克写个脚本交给赫尔墨斯优化")
    assert r.collab is True
    assert r.collab_mode == "handoff"


def test_review_direction_by_utterance(router):
    # "让小樱评审赫尔墨斯" -> 小樱 reviews 赫尔墨斯; utterance order [cherrystudio, hermes]
    r = router.route("让小樱评审赫尔墨斯")
    assert r.collab_mode == "review"
    assert r.collab_agents == ["cherrystudio", "hermes"]


def test_handoff_direction_by_utterance(router):
    # reversed phrasing must reverse the pipeline order
    r = router.route("赫尔墨斯先写然后让小克优化")
    assert r.collab_mode == "handoff"
    assert r.collab_agents == ["hermes", "openclaw"]


def test_active_is_per_connection(router):
    # Two simultaneous voice calls share the single Router instance; passing
    # `active` must keep their active-agent state isolated.
    a = router.default  # caller A's active agent
    b = router.default  # caller B's active agent
    r_b = router.route("赫尔墨斯，写个总结", active=b)
    b = r_b.agent
    assert b == "hermes"  # B switched to hermes
    # A's no-alias utterance must stay on default, NOT inherit B's hermes.
    r_a = router.route("今天天气怎么样", active=a)
    assert r_a.agent == router.default
    # Shared self.active must not be polluted by per-call active states.
    assert router.active == router.default
