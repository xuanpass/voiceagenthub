"""声纹注册表：在线聚类 + 临时簇攒段转正 + JSON 持久化。

纯逻辑模块——不 import pipecat / sherpa，便于单测直接跑。

聚类策略（阈值来自实测：同人 cos≈0.96，不同段 cos≈0.635）：
1. 先对比已命名声纹（named_threshold=0.75）→ 命中即身份；
2. 再对比临时簇质心（cluster_threshold=0.70）→ EMA(alpha=0.3) 归并，
   攒够 min_segments 段即"发现新声音"，提示用户命名；
3. 都不中 → 新开临时簇 vc<hex6>。

临时簇只在内存中跨通话存活（设计上不落盘），命名后才持久化——
避免无人认领的匿名簇越积越多、以及同音误聚被永久固化。
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid

logger = logging.getLogger("voicehub")


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度；任一零范数返回 0。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _l2_norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n == 0.0:
        return list(v)
    return [x / n for x in v]


class VoiceprintRegistry:
    """命名声纹 + 临时簇的在线聚类注册表（asyncio 单线程使用，无锁）。"""

    def __init__(
        self,
        named_threshold: float = 0.75,
        cluster_threshold: float = 0.70,
        min_segments: int = 3,
        persist_path: str | None = None,
        ema_alpha: float = 0.3,
    ):
        self.named_threshold = named_threshold
        self.cluster_threshold = cluster_threshold
        self.min_segments = min_segments
        self.ema_alpha = ema_alpha
        self._persist_path = persist_path
        # name -> {"centroid": [..512], "segments": int, "created_at": float}
        self.named: dict[str, dict] = {}
        # cluster_id -> {"centroid": [..512], "segments": int, "last_seen": float}
        self.tentative: dict[str, dict] = {}
        if persist_path:
            self._load()

    # ---------- 核心：分类/归并 ----------

    def classify(self, emb: list[float]) -> dict:
        """把一段声纹映射到命名身份 / 临时簇 / 新簇。

        返回 {"label", "confidence", "enrolled", "cluster_id",
              "is_new_cluster", "ready_now", "segments"}。
        ready_now: 本次更新后临时簇恰好攒到 min_segments（触发命名提示）。
        """
        now = time.time()
        emb = _l2_norm(emb)

        # 1) 已命名声纹（最高相似度优先）
        best_name, best_cos = None, 0.0
        for name, rec in self.named.items():
            c = cosine(emb, rec["centroid"])
            if c > best_cos:
                best_name, best_cos = name, c
        if best_name is not None and best_cos >= self.named_threshold:
            return {
                "label": best_name, "confidence": round(best_cos, 3),
                "enrolled": True, "cluster_id": None,
                "is_new_cluster": False, "ready_now": False,
                "segments": self.named[best_name]["segments"],
            }

        # 2) 临时簇归并（EMA 更新质心）
        best_id, best_cc = None, 0.0
        for cid, rec in self.tentative.items():
            c = cosine(emb, rec["centroid"])
            if c > best_cc:
                best_id, best_cc = cid, c
        if best_id is not None and best_cc >= self.cluster_threshold:
            rec = self.tentative[best_id]
            mixed = [
                self.ema_alpha * e + (1.0 - self.ema_alpha) * o
                for o, e in zip(rec["centroid"], emb)
            ]
            rec["centroid"] = _l2_norm(mixed)
            rec["segments"] += 1
            rec["last_seen"] = now
            ready = rec["segments"] == self.min_segments
            return {
                "label": best_id, "confidence": round(best_cc, 3),
                "enrolled": False, "cluster_id": best_id,
                "is_new_cluster": False, "ready_now": ready,
                "segments": rec["segments"],
            }

        # 3) 新临时簇
        cid = "vc" + uuid.uuid4().hex[:6]
        self.tentative[cid] = {"centroid": emb, "segments": 1, "last_seen": now}
        return {
            "label": cid, "confidence": 1.0,
            "enrolled": False, "cluster_id": cid,
            "is_new_cluster": True, "ready_now": False, "segments": 1,
        }

    def cluster_ready(self, cluster_id: str) -> bool:
        rec = self.tentative.get(cluster_id)
        return bool(rec and rec["segments"] >= self.min_segments)

    # ---------- 转正 / 管理 ----------

    @staticmethod
    def _validate_name(name: str, named: dict[str, dict]) -> str:
        name = (name or "").strip()
        if not name:
            raise ValueError("name empty")
        if name.startswith("vc"):
            raise ValueError("name prefix 'vc' reserved for clusters")
        if name in named:
            raise ValueError(f"name exists: {name}")
        return name

    def enroll_from_cluster(self, cluster_id: str, name: str) -> None:
        """临时簇 → 命名声纹（持久化）。"""
        rec = self.tentative.get(cluster_id)
        if rec is None:
            raise KeyError(f"unknown cluster: {cluster_id}")
        name = self._validate_name(name, self.named)
        self.named[name] = {
            "centroid": list(rec["centroid"]),
            "segments": rec["segments"],
            "created_at": time.time(),
        }
        del self.tentative[cluster_id]
        self._save()
        logger.warning("[PROBE-VOICEPRINT] enrolled name=%s from=%s segs=%s",
                       name, cluster_id, rec["segments"])

    def enroll(self, name: str, embs: list[list[float]]) -> None:
        """直接以多段声纹注册命名身份（API 录音注册路径）。"""
        if not embs:
            raise ValueError("no embeddings")
        name = self._validate_name(name, self.named)
        mean = [sum(v) for v in zip(*embs)]
        self.named[name] = {
            "centroid": _l2_norm(mean),
            "segments": len(embs),
            "created_at": time.time(),
        }
        self._save()

    def merge(self, source: str, target: str) -> None:
        """合并两个命名声纹（防簇分裂：同人被分成两个名字时人工合一）。"""
        if source not in self.named:
            raise KeyError(f"unknown name: {source}")
        if target not in self.named:
            raise KeyError(f"unknown name: {target}")
        if source == target:
            raise ValueError("source == target")
        a = self.named[source]["centroid"]
        b = self.named[target]["centroid"]
        merged = _l2_norm([(x + y) / 2.0 for x, y in zip(a, b)])
        segs = self.named[source]["segments"] + self.named[target]["segments"]
        self.named[target]["centroid"] = merged
        self.named[target]["segments"] = segs
        del self.named[source]
        self._save()

    def delete(self, target: str) -> str:
        """删除命名声纹或临时簇，返回被删类型。"""
        if target in self.named:
            del self.named[target]
            self._save()
            return "named"
        if target in self.tentative:
            del self.tentative[target]
            self._save()
            return "tentative"
        raise KeyError(f"unknown target: {target}")

    def clear(self) -> None:
        self.named.clear()
        self.tentative.clear()
        self._save()

    def snapshot(self) -> dict:
        """API 视图：不暴露 512 维原始质心。"""
        return {
            "named": [
                {"name": n, "segments": r["segments"],
                 "created_at": r["created_at"], "dim": len(r["centroid"])}
                for n, r in sorted(self.named.items())
            ],
            "tentative": [
                {"cluster_id": c, "segments": r["segments"],
                 "last_seen": r["last_seen"]}
                for c, r in sorted(self.tentative.items())
            ],
        }

    # ---------- 持久化（只存命名声纹） ----------

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            data = {
                "named": {
                    n: {"centroid": r["centroid"], "segments": r["segments"],
                        "created_at": r["created_at"]}
                    for n, r in self.named.items()
                }
            }
            tmp = self._persist_path + ".tmp"
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self._persist_path)
        except Exception as e:
            logger.warning("[PROBE-VOICEPRINT] save err %r", e)

    def _load(self) -> None:
        try:
            if not os.path.isfile(self._persist_path):
                return
            with open(self._persist_path, encoding="utf-8") as f:
                data = json.load(f)
            for name, rec in (data.get("named") or {}).items():
                if rec.get("centroid"):
                    self.named[name] = {
                        "centroid": [float(x) for x in rec["centroid"]],
                        "segments": int(rec.get("segments", 1)),
                        "created_at": float(rec.get("created_at", 0.0)),
                    }
            if self.named:
                logger.warning("[PROBE-VOICEPRINT] loaded %d voiceprints", len(self.named))
        except Exception as e:
            logger.warning("[PROBE-VOICEPRINT] load err %r", e)
            self.named.clear()
