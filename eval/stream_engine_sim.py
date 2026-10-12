"""Two-pass streaming engine prototype, simulated in real time on CPU.

A "call" = several dev utterances joined with ~1 s pauses. Per call:
  VAD (Silero vad.onnx from the model repo, $VAD_ONNX; 96 ms windows): speech start after >=250 ms of speech, endpoint after
    >=500 ms of silence, 200 ms padding on both sides.
  Partials: every CHUNK ms of speech, the stage-C model decodes the segment so far under the chunk mask (prefix
    recompute: the same output a cache-aware engine produces, just slower) and shows the completed chunks' text.
  Final: at each endpoint, the full-context fine-tuned model decodes the segment (second pass).
Measures, per partial word: lag = time the word first appears in a partial and never changes again, minus the time
the speaker finished it (from the full-context model's CTC frames). Also: final latency after end of speech
(endpoint wait + final decode compute), partial rewrite rate, and final WER vs offline decoding of the same calls.
usage: stream_engine_sim.py OUT.json MANIFEST [MANIFEST ...]   (env: CHUNK_FRAMES=16, CALLS=20, PROCS=8)
"""
import json, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "service")]
SR, WIN, FRAME = 16000, 1536, 0.04
VAD = os.environ.get("VAD", "repo")          # repo (the model repo's vad.onnx + AGC) | silero (Silero VAD v5, MIT)
PAD = float(os.environ.get("PAD", "0.2"))    # padding added on both sides of a speech segment (s)
EOU = float(os.environ.get("EOU", "0.5"))    # silence that ends an utterance (s); pauses shorter than this don't split
C = int(os.environ.get("CHUNK_FRAMES", "16"))
SEG = os.environ.get("SEG", "vad")           # vad | oracle (true utterance spans +- PAD: isolates VAD errors)
FILL_DB = os.environ.get("FILL_DB") or None  # pauses between utterances: digital zeros (unset) or white noise at this dBFS RMS
NO_PARTIALS = os.environ.get("NO_PARTIALS") == "1"  # finals only (fast diagnostics)
VAD_THR = float(os.environ.get("VAD_THR", "0.5"))    # speech probability threshold
ENGINE = os.environ.get("ENGINE", "redecode")    # partials: redecode (prefix recompute) | cache (cache_engine.py)
THREADS = int(os.environ.get("THREADS", "4"))   # torch threads per worker (keep workers x threads <= cores)
BF16 = os.environ.get("BF16") == "1"                 # CPU bf16 autocast (AMX) for both models
SHOW_LAST = os.environ.get("SHOW_LAST") == "1"  # show the still-growing last word tentatively (lower lag, some rewrites)
# MERGE=1 (default): padded segments that overlap are merged, which needs the future (the first final would already
# have been emitted live). MERGE=0: live-realistic: each segment's pre-roll is clamped to start where the previous
# padded segment ended, so no audio is decoded twice and no final is revised.
MERGE = os.environ.get("MERGE", "1") == "1"
C_CKPT = os.environ.get("C_CKPT", str(STT / "runs/stream_multilingual_large_ctc/final_stream.ckpt"))
F_CKPT = os.environ.get("F_CKPT", str(STT / "runs/full_multilingual_large_ctc/final_a1.00.ckpt"))
M = {}


def init():
    import torch, gigaam, onnxruntime as ort
    import streaming_patch
    torch.set_num_threads(THREADS)
    gigaam._CACHE_DIR = str(STT / ".gigaam-cache")
    streaming_patch.enable()
    M["c"] = gigaam.load_model(C_CKPT, fp16_encoder=False, device="cpu").eval()
    M["f"] = gigaam.load_model(F_CKPT, fp16_encoder=False, device="cpu").eval()
    M["f"].encoder._force_full = True
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
    M["vad"] = ort.InferenceSession(os.environ.get("VAD_ONNX", str(STT / "models/old/vad/vad.onnx")), o)
    if VAD == "silero":
        from silero_vad import load_silero_vad
        M["silero"] = load_silero_vad(onnx=True)
    M["vocab"] = M["f"].decoding.tokenizer.vocab if hasattr(M["f"].decoding.tokenizer, "vocab") else None


def logprobs(model, wav, chunk):
    import torch
    os.environ["CHUNK_EVAL"], os.environ["LOOKAHEAD_EVAL"] = str(chunk), "0"
    x = torch.from_numpy(wav).unsqueeze(0)
    with torch.inference_mode():
        feats, fl = model.preprocessor(x, torch.tensor([x.shape[1]]))  # mel in fp32 always
        with torch.autocast("cpu", torch.bfloat16, enabled=BF16):
            enc, _ = model.encoder(feats, fl)
            return model.head(enc)[0].float().numpy()


def words_with_end(lp, vocab, offset):
    """Greedy CTC words with the time their last character was emitted."""
    blank, out, cur, last_t, prev = len(vocab), [], "", 0.0, -1
    for f, i in enumerate(lp.argmax(-1).tolist()):
        if i != prev and i != blank:
            ch = vocab[i]
            if ch == " ":
                if cur:
                    out.append((cur, last_t))
                cur = ""
            else:
                cur += ch; last_t = offset + (f + 1) * FRAME
        prev = i
    if cur:
        out.append((cur, last_t))
    return out


def vad_probs(wav):
    """Per-window speech probability, window length in seconds."""
    if VAD == "silero":
        import torch
        M["silero"].reset_states()
        n = 512  # Silero v5 window at 16 kHz (32 ms)
        x = torch.from_numpy(wav)
        return [float(M["silero"](x[i:i + n], SR)) for i in range(0, len(wav) - n + 1, n)], n / SR
    # old repo VAD on an AGC'd copy: it misses very quiet recordings (peak ~-36 dBFS -> speech prob < 0.3) that the
    # ASR transcribes fine. Gain follows a running RMS (target -20 dBFS, max x30), fast attack / slow release.
    h = c = np.zeros((2, 1, 64), np.float32)
    probs, rms = [], None
    for i in range(0, len(wav) - WIN + 1, WIN):
        x = wav[i:i + WIN]
        r = float(np.sqrt(np.mean(x ** 2)) + 1e-6)
        rms = r if rms is None else (0.9 * rms + 0.1 * r if r < rms else 0.5 * rms + 0.5 * r)
        x = np.clip(x * min(30.0, 0.1 / rms), -1, 1).astype(np.float32)
        p, h, c = M["vad"].run(None, {"input": x[None, :], "sr": np.array(SR, np.int64), "h": h, "c": c})
        probs.append(float(np.ravel(p)[0]))
    return probs, WIN / SR


def vad_segments(wav):
    """Speech segments (start, end, time the endpoint is known). Start after >=250 ms of speech; end after EOU s of
    silence; PAD s of padding on both sides."""
    probs, w = vad_probs(wav)
    segs, start, sil, run = [], None, 0, 0
    for k, p in enumerate(probs):
        t = k * w
        if p > VAD_THR:
            run += 1; sil = 0
            if start is None and run * w >= 0.25:
                start = t - (run - 1) * w
        else:
            run = 0
            if start is not None:
                sil += 1
                if sil * w >= EOU:
                    end = t - (sil - 1) * w
                    segs.append((max(0.0, start - PAD), min(len(wav) / SR, end + PAD), t + w))
                    start, sil = None, 0
    if start is not None:
        segs.append((max(0.0, start - PAD), len(wav) / SR, len(wav) / SR))
    if not MERGE:  # live: clamp the pre-roll to the previous segment's end
        out = []
        for a, b, e in segs:
            a = max(a, out[-1][1]) if out else a
            if b > a:
                out.append((a, b, e))
        return out
    # padding can make neighbours overlap: merge them so no audio is decoded twice
    merged = []
    for a, b, e in segs:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]), e)
        else:
            merged.append((a, b, e))
    return merged


def run_call(call):
    vocab = [str(v) for v in M["f"].decoding.tokenizer.vocab]  # OmegaConf ListConfig rejects numpy int indices
    wav = np.concatenate(call["wavs"]).astype(np.float32)
    finals, lags, rewrites, shown, final_lat, part_compute, n_part = [], [], 0, 0, [], [], 0
    if SEG == "oracle":
        segs = [(max(0.0, a - PAD), min(len(wav) / SR, b + PAD), min(len(wav) / SR, b + EOU)) for a, b in call["spans"]]
    else:
        segs = vad_segments(wav)
    for s0, s1, t_eou in segs:
        seg = wav[int(s0 * SR):int(s1 * SR)]
        if NO_PARTIALS:
            t0 = time.perf_counter()
            finals += [w for w, _ in words_with_end(logprobs(M["f"], seg, 0), vocab, s0)]
            final_lat.append((t_eou - (s1 - PAD)) + (time.perf_counter() - t0))
            continue
        # full-context alignment of this segment = when each word was actually finished
        truth = words_with_end(logprobs(M["f"], seg, 0), vocab, s0)
        first_seen, history = {}, []
        if ENGINE == "cache":
            # cache-aware engine (cache_engine.py): chunk k is computed once, when its audio + 40 ms lookahead has
            # arrived; if compute is slower than real time, chunks queue (emit = max(ready, previous emit) + compute)
            from cache_engine import CacheEngine
            eng, lps, t_free, pos, k = CacheEngine(M["c"], C, cpu_bf16=BF16), [], 0.0, 0, 0
            la = int(round(eng.lookahead_ms * SR / 1000))
            while True:
                ready = (k + 1) * C * 640 + la
                at_end = ready >= len(seg)
                t0 = time.perf_counter()
                new = eng.feed(seg[pos:ready]) if not at_end else eng.feed(seg[pos:]) + eng.finish()
                dt = time.perf_counter() - t0
                pos = min(ready, len(seg))
                t_emit = max(pos / SR, t_free) + dt; t_free = t_emit
                part_compute.append(dt); n_part += 1
                lps += [x.float().numpy() for x in new]
                lp = np.concatenate(lps) if lps else np.zeros((0, len(vocab) + 1), np.float32)
                words = [w for w, _ in words_with_end(lp, vocab, s0)]
                if words and not at_end and not SHOW_LAST:
                    words = words[:-1]
                for i, w in enumerate(words):
                    if first_seen.get(i, (None,))[0] != w:
                        first_seen[i] = (w, s0 + t_emit)
                history.append(words)
                if at_end:
                    break
                k += 1
        else:
            step = C * FRAME
            for k in range(1, int(len(seg) / SR / step) + 2):
                t_audio = min(k * step, len(seg) / SR)
                t0 = time.perf_counter()
                lp = logprobs(M["c"], seg[:int(t_audio * SR)], C)
                dt = time.perf_counter() - t0
                part_compute.append(dt); n_part += 1
                at_end = t_audio >= len(seg) / SR
                n_done = lp.shape[0] if at_end else min(lp.shape[0], int(round(t_audio / FRAME)) // C * C)
                words = [w for w, _ in words_with_end(lp[:n_done], vocab, s0)]
                if words and not at_end and not SHOW_LAST:
                    words = words[:-1]  # the last word may still be growing into the next chunk: hide it
                for i, w in enumerate(words):
                    if first_seen.get(i, (None,))[0] != w:
                        first_seen[i] = (w, s0 + t_audio + dt)
                history.append(words)
        for prev in history[:-1]:
            shown += len(prev)
            rewrites += sum(1 for i, w in enumerate(prev) if i >= len(history[-1]) or history[-1][i] != w)
        t0 = time.perf_counter()
        final_words = words_with_end(logprobs(M["f"], seg, 0), vocab, s0)
        final_lat.append((t_eou - (s1 - PAD)) + (time.perf_counter() - t0))
        finals += [w for w, _ in final_words]
        for i, (w, t_end) in enumerate(truth):
            if i in first_seen and first_seen[i][0] == w:
                lags.append(first_seen[i][1] - t_end)
    offline = []
    for wv in call["wavs"][0::2]:  # the utterances alone (odd entries are the pauses)
        offline += [w for w, _ in words_with_end(logprobs(M["f"], wv.astype(np.float32), 0), vocab, 0)]
    return {"segments": [(round(a, 2), round(b, 2)) for a, b, _ in segs], "spans": call["spans"],
            "ref": call["ref"], "final": " ".join(finals), "offline": " ".join(offline), "lags": lags,
            "final_latency": final_lat, "rewrites": rewrites, "shown": shown, "partial_compute": part_compute,
            "audio_s": len(wav) / SR}


def load_calls(manifests, n_calls, per_call=3):
    import soundfile as sf
    from textnorm import normalize
    rows = [json.loads(l) for m in manifests for l in open(m, encoding="utf-8")]
    random.Random(1).shuffle(rows)
    calls = []
    for i in range(n_calls):
        grp = rows[i * per_call:(i + 1) * per_call]
        wavs, refs, spans, t = [], [], [], 0.0
        for r in grp:
            w, sr = sf.read(r["path"], dtype="float32")
            w = w if w.ndim == 1 else w.mean(1)
            n = int(SR * random.Random(i).uniform(0.9, 1.3))
            gap = np.zeros(n, np.float32) if FILL_DB is None else \
                np.random.default_rng(i).normal(0, 10 ** (float(FILL_DB) / 20), n).astype(np.float32)
            wavs += [w, gap]
            spans.append((t, t + len(w) / SR)); t += (len(w) + n) / SR
            refs.append(normalize(r["ref"]))
        calls.append({"wavs": wavs, "ref": " ".join(refs), "spans": spans})
    return calls


if __name__ == "__main__":
    import jiwer
    from textnorm import normalize
    out, manifests = sys.argv[1], sys.argv[2:]
    calls = load_calls(manifests, int(os.environ.get("CALLS", "20")))
    with ProcessPoolExecutor(int(os.environ.get("PROCS", "8")), initializer=init) as ex:
        res = list(ex.map(run_call, calls))
    lags = [l for r in res for l in r["lags"]] or [float("nan")]
    fl = [l for r in res for l in r["final_latency"]] or [float("nan")]
    pc = [c for r in res for c in r["partial_compute"]] or [float("nan")]
    refs = [r["ref"] for r in res]
    summary = {
        "chunk_ms": C * 40, "vad": VAD, "vad_thr": VAD_THR, "engine": ENGINE, "show_last": SHOW_LAST, "merge": MERGE, "bf16": BF16, "seg": SEG, "fill_db": FILL_DB, "threads": THREADS, "pad_s": PAD, "eou_s": EOU, "calls": len(res), "audio_min": round(sum(r["audio_s"] for r in res) / 60, 1),
        "partial_word_lag_ms": {"mean": round(1000 * float(np.mean(lags)), 0), "p50": round(1000 * float(np.median(lags)), 0),
                                "p90": round(1000 * float(np.percentile(lags, 90)), 0)},
        "final_latency_after_speech_end_ms": {"mean": round(1000 * float(np.mean(fl)), 0), "p90": round(1000 * float(np.percentile(fl, 90)), 0)},
        "partial_rewrite_rate": round(sum(r["rewrites"] for r in res) / max(1, sum(r["shown"] for r in res)), 4),
        "partial_compute_s_per_update": {"mean": round(float(np.mean(pc)), 3), "p90": round(float(np.percentile(pc, 90)), 3)},
        "final_wer": round(100 * jiwer.wer(refs, [normalize(r["final"]) for r in res]), 2),
        "offline_wer_same_calls": round(100 * jiwer.wer(refs, [normalize(r["offline"]) for r in res]), 2),
    }
    summary["per_call"] = [{"ref": r["ref"], "final": normalize(r["final"]), "offline": normalize(r["offline"]),
                            "segments": r.get("segments")} for r in res]
    json.dump(summary, open(out, "w"), indent=1, ensure_ascii=False)
    summary.pop("per_call")
    print(json.dumps(summary, indent=1))
