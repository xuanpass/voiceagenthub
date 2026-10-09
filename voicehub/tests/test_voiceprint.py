"""声纹子系统单元测试（脚本式，与 test_gate_pipeline.py 同风格）。

覆盖：
- registry: 新簇/归并/攒段转正/命名校验/持久化/合并删除清空
- processor: 事件推送/转发/短段跳过/回声跳过/speaker_found 单次去重
- embedder: 模型缺失时优雅降级（不抛异常）

运行: .venv/bin/python voicehub/tests/test_voiceprint.py
"""
import asyncio
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipecat.frames.frames import (
    InputAudioRawFrame,
    TTSAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from server.voiceprint import SpeakerEmbedder, VoiceprintRegistry, SpeakerIdProcessor, cosine


# ---------- 工具 ----------

def make_emb(seed: int, dim: int = 64) -> list[float]:
    """确定性伪随机单位向量（避免依赖 numpy）。"""
    v = [((seed * 1103515245 + i * 12345) % 10007) / 10007.0 - 0.5 for i in range(dim)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def perturb(emb: list[float], amt: float) -> list[float]:
    """微扰：保持高余弦（同人不同段）。"""
    v = [x + amt * ((i * 7919) % 100) / 100.0 - amt / 2 for i, x in enumerate(emb)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def pcm_bytes(seconds: float, sr: int = 16000) -> bytes:
    return bytes(int(seconds * sr) * 2)  # 全零 int16 PCM


class FakeEmbedder:
    """按调用顺序返回预设向量；enabled=True。"""

    def __init__(self, vectors):
        self._vectors = list(vectors)
        self.enabled = True
        self.calls = 0

    async def embed(self, pcm_bytes, sample_rate=16000):
        self.calls += 1
        if not self._vectors:
            return None
        return self._vectors.pop(0)


class CaptureSpeaker(SpeakerIdProcessor):
    """记录 push_frame 与 send_msg。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pushed = []
        self.sent = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append(frame)

    async def _capture_send(self, msg):
        self.sent.append(msg)


async def drive_utterance(proc, seconds: float, chunk_s: float = 0.6, bot_audio=False):
    """喂一整段发言：VAD 起 → N 个输入帧 → VAD 止（可选夹杂机器人播报）。"""
    await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    remaining = seconds
    while remaining > 0:
        d = min(chunk_s, remaining)
        await proc.process_frame(
            InputAudioRawFrame(audio=pcm_bytes(d), sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        if bot_audio:
            await proc.process_frame(
                TTSAudioRawFrame(audio=pcm_bytes(0.1), sample_rate=16000, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )
        remaining -= d
    await proc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)


async def main():
    results = {}

    # ===== registry =====

    # 1) 新段 → 新临时簇
    reg = VoiceprintRegistry(min_segments=3)
    e1 = make_emb(1)
    r = reg.classify(e1)
    results["registry_new_cluster"] = (
        r["is_new_cluster"] and r["label"].startswith("vc")
        and r["cluster_id"] == r["label"] and not r["enrolled"]
    )

    # 2) 微扰段 → 归并同簇，segments=2
    r2 = reg.classify(perturb(e1, 0.05))
    results["registry_same_cluster_merge"] = (
        not r2["is_new_cluster"] and r2["cluster_id"] == r["cluster_id"]
        and r2["segments"] == 2
    )

    # 3) 第 3 段 → ready_now=True 且 cluster_ready
    r3 = reg.classify(perturb(e1, 0.06))
    results["registry_ready_after_min"] = (
        r3["segments"] == 3 and r3["ready_now"] and reg.cluster_ready(r3["cluster_id"])
    )

    # 4) 第 4 段 → ready_now 不再触发（只报一次）
    r4 = reg.classify(perturb(e1, 0.07))
    results["registry_ready_once"] = (r4["segments"] == 4 and not r4["ready_now"])

    # 5) 不相似段 → 新簇（阈值下不误并）
    r_other = reg.classify(make_emb(99))
    results["registry_split_dissimilar"] = (
        r_other["is_new_cluster"] and r_other["cluster_id"] != r["cluster_id"]
    )

    # 6) 转正：命名后 classify 命中 enrolled
    reg.enroll_from_cluster(r["cluster_id"], "小明")
    r5 = reg.classify(perturb(e1, 0.05))
    results["registry_enroll_promote"] = (
        r5["enrolled"] and r5["label"] == "小明" and r["cluster_id"] not in reg.tentative
    )
    # 命名身份余弦应高于未命名阈值
    results["registry_enrolled_cos_high"] = r5["confidence"] >= reg.named_threshold

    # 7) 命名校验：重复/空/vc前缀/缺簇
    try:
        reg.enroll_from_cluster(r["cluster_id"], "小明")  # 簇已消费
        results["registry_missing_cluster_raises"] = False
    except KeyError:
        results["registry_missing_cluster_raises"] = True
    reg2 = VoiceprintRegistry()
    e_b = make_emb(2)
    c_b = reg2.classify(e_b)["cluster_id"]
    reg2.enroll_from_cluster(c_b, "甲")
    try:
        reg2.enroll_from_cluster(c_b, "乙")
        results["registry_missing_cluster2_raises"] = False
    except KeyError:
        results["registry_missing_cluster2_raises"] = True
    try:
        reg2.classify(e_b)  # 新建一个临时簇用于测试非法名
        pass
    except Exception:
        pass
    c_c = reg2.classify(make_emb(3))["cluster_id"]
    for bad in ("  ", "vc123456"):
        try:
            reg2.enroll_from_cluster(c_c, bad)
            results[f"registry_bad_name_{bad.strip() or 'empty'}_raises"] = False
        except ValueError:
            results[f"registry_bad_name_{bad.strip() or 'empty'}_raises"] = True
    # 重复名（若簇还在则先转正）
    c_d = reg2.classify(make_emb(4))["cluster_id"]
    try:
        reg2.enroll_from_cluster(c_d, "甲")
        results["registry_dup_name_raises"] = False
    except ValueError:
        results["registry_dup_name_raises"] = True

    # 8) 持久化往返
    tmpdir = tempfile.mkdtemp(prefix="vp-test-")
    try:
        path = os.path.join(tmpdir, "voiceprints.json")
        regA = VoiceprintRegistry(persist_path=path)
        cA = regA.classify(make_emb(7))["cluster_id"]
        regA.enroll_from_cluster(cA, "老王")
        regB = VoiceprintRegistry(persist_path=path)
        results["registry_persistence_roundtrip"] = "老王" in regB.named
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # 9) 合并/删除/清空
    regM = VoiceprintRegistry()
    c1 = regM.classify(make_emb(11))["cluster_id"]
    regM.enroll_from_cluster(c1, "A")
    c2 = regM.classify(make_emb(22))["cluster_id"]
    regM.enroll_from_cluster(c2, "B")
    regM.merge("A", "B")
    results["registry_merge"] = ("A" not in regM.named and "B" in regM.named
                                 and regM.named["B"]["segments"] == 2)
    kind = regM.delete("B")
    results["registry_delete"] = (kind == "named" and not regM.named)
    c3 = regM.classify(make_emb(33))["cluster_id"]
    kind2 = regM.delete(c3)
    results["registry_delete_tentative"] = (kind2 == "tentative" and not regM.tentative)
    regM.classify(make_emb(44))
    regM.clear()
    snap = regM.snapshot()
    results["registry_clear"] = (not snap["named"] and not snap["tentative"])

    # 10) snapshot 不泄漏质心
    regS = VoiceprintRegistry()
    cS = regS.classify(make_emb(55))["cluster_id"]
    regS.enroll_from_cluster(cS, "C")
    snap2 = regS.snapshot()
    results["snapshot_no_centroid"] = all(
        "centroid" not in x for x in snap2["named"] + snap2["tentative"]
    )

    # 11) 命名命中阈值边界：与命名向量差异过大 → 不命中，落新簇
    regT = VoiceprintRegistry(named_threshold=0.75)
    eT = make_emb(66)
    cT = regT.classify(eT)["cluster_id"]
    regT.enroll_from_cluster(cT, "远")
    r_far = regT.classify(make_emb(77))  # 完全不同的向量 cos≈0
    results["registry_no_false_match"] = not r_far["enrolled"]

    # ===== embedder =====

    # 12) 模型缺失 → 优雅 None
    emb_missing = SpeakerEmbedder(model_path="/nonexistent/model.onnx")
    got = await emb_missing.embed(pcm_bytes(2.0))
    results["embedder_disabled_graceful"] = (emb_missing.enabled is False and got is None)

    # 13) 短段在模型缺失场景也走门控（不会先崩）——逻辑顺序验证
    results["embedder_short_segment_none"] = (await emb_missing.embed(pcm_bytes(0.5))) is None

    # ===== processor =====

    vecs = [make_emb(1), perturb(make_emb(1), 0.05), perturb(make_emb(1), 0.06)]
    fake = FakeEmbedder(vecs)
    regP = VoiceprintRegistry(min_segments=3)
    proc = CaptureSpeaker(fake, regP, "conn-test", None, min_s=1.0)
    proc._send_msg = proc._capture_send

    # 14) 正常发言 1.2s → 至少推 1 条 speaker 事件，且所有帧被转发
    await drive_utterance(proc, 1.2)
    await asyncio.sleep(0.05)
    n_pushed = len(proc.pushed)
    results["processor_emits_speaker_event"] = (
        any(m.get("type") == "speaker" for m in proc.sent)
        and n_pushed == 5  # 1 start + 2 input + 1 stop... 实为 1+2+1=4? 见下断言
    )
    # 修正：1 start + 2 input + 1 stop = 4 帧
    results["processor_emits_speaker_event"] = (
        any(m.get("type") == "speaker" for m in proc.sent) and n_pushed == 4
    )
    results["processor_forwards_frames"] = n_pushed == 4

    # 15) 短段（0.6s < 1.0s min_s）→ 不发事件
    proc2 = CaptureSpeaker(FakeEmbedder([make_emb(2)]), VoiceprintRegistry(),
                           "c2", None, min_s=1.0)
    proc2._send_msg = proc2._capture_send
    await drive_utterance(proc2, 0.6, chunk_s=0.6)
    await asyncio.sleep(0.05)
    results["processor_skips_short"] = not any(
        m.get("type") == "speaker" for m in proc2.sent
    )

    # 16) 回声窗口内（缓冲期间有 TTS 帧）→ 不发事件
    proc3 = CaptureSpeaker(FakeEmbedder([make_emb(3)]), VoiceprintRegistry(),
                           "c3", None, min_s=1.0)
    proc3._send_msg = proc3._capture_send
    await drive_utterance(proc3, 1.2, bot_audio=True)
    await asyncio.sleep(0.05)
    results["processor_skips_echo"] = not any(
        m.get("type") == "speaker" for m in proc3.sent
    )

    # 17) 回声段跳过后、下一段干净 → 正常出事件（回声门按段重置）
    await drive_utterance(proc3, 1.2, bot_audio=False)
    await asyncio.sleep(0.05)
    results["processor_echo_gate_resets"] = any(
        m.get("type") == "speaker" for m in proc3.sent
    )

    # 18) 攒段 → speaker_found 只推一次
    fake4 = FakeEmbedder([make_emb(5), perturb(make_emb(5), 0.05), perturb(make_emb(5), 0.06)])
    reg4 = VoiceprintRegistry(min_segments=3)
    proc4 = CaptureSpeaker(fake4, reg4, "c4", None, min_s=1.0)
    proc4._send_msg = proc4._capture_send
    for _ in range(3):
        await drive_utterance(proc4, 1.2)
        await asyncio.sleep(0.05)
    found = [m for m in proc4.sent if m.get("type") == "speaker_found"]
    # 第 4 段（若再喂）也不重复；这里先断言恰好 1 条
    results["processor_speaker_found_once"] = (len(found) == 1 and found[0]["cluster_id"].startswith("vc"))

    # 19) 同标签冷却：连续两段同人 → 第二段不重复推 speaker（5s 内）
    n_speaker_before = len([m for m in proc4.sent if m.get("type") == "speaker"])
    fake4._vectors.append(perturb(make_emb(5), 0.05))
    await drive_utterance(proc4, 1.2)
    await asyncio.sleep(0.05)
    n_speaker_after = len([m for m in proc4.sent if m.get("type") == "speaker"])
    results["processor_speaker_dedupe"] = (n_speaker_after == n_speaker_before)

    # 20) embed 异常不炸管线：FakeEmbedder 抛错 → 仍转发、无异常冒出
    class BoomEmbedder:
        enabled = True
        async def embed(self, *a, **k):
            raise RuntimeError("boom")
    proc5 = CaptureSpeaker(BoomEmbedder(), VoiceprintRegistry(), "c5", None, min_s=1.0)
    proc5._send_msg = proc5._capture_send
    try:
        await drive_utterance(proc5, 1.2)
        await asyncio.sleep(0.05)
        results["processor_survives_embed_error"] = (len(proc5.pushed) == 4)
    except Exception:
        results["processor_survives_embed_error"] = False

    return results


if __name__ == "__main__":
    results = asyncio.run(main())
    print("\n=== Voiceprint Test Results ===")
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}")
    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
