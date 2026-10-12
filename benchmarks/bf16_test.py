"""AMX bf16 vs fp32 on CPU for our models: latency and WER parity.
Part A (one process, threads 1 and 4, fp32/bf16 interleaved): cache-engine time per chunk (C v1, 320/640 ms) on 6 dev
  utterances, and the final-pass decode (offline FT) of 5/10/20 s segments; plus text identity bf16 vs fp32.
Part B (pool of 32 x 1 thread): greedy WER on dev sets (FLEURS kk/ru, KSC2 spont; 100 utterances each) for C v1 at
  320/640 ms (= engine partial text) and the offline FT, fp32 vs bf16.
usage: bf16_test.py OUT.json"""
import json, os, platform, random, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "service")]
C_CKPT = os.environ.get("C_CKPT", str(STT / "runs/stream_multilingual_large_ctc/final_stream.ckpt"))
F_CKPT = os.environ.get("F_CKPT", str(STT / "runs/full_multilingual_large_ctc/final_a1.00.ckpt"))
SETS = ["dev_fleurs_kk", "dev_fleurs_ru", "dev_ksc2_spont"]
M = {}


def load_models():
    import gigaam, streaming_patch
    gigaam._CACHE_DIR = str(STT / ".gigaam-cache")
    streaming_patch.enable()
    M["c"] = gigaam.load_model(C_CKPT, fp16_encoder=False, device="cpu").eval()
    M["f"] = gigaam.load_model(F_CKPT, fp16_encoder=False, device="cpu").eval()
    M["f"].encoder._force_full = True
    M["vocab"] = [str(v) for v in M["f"].decoding.tokenizer.vocab]


def greedy(lp):
    out, prev = [], -1
    for i in lp.argmax(-1).tolist():
        if i != prev and i != len(M["vocab"]):
            out.append(M["vocab"][i])
        prev = i
    return " ".join("".join(out).split())


def lp_masked(model, wav, C, bf16):
    os.environ["CHUNK_EVAL"], os.environ["LOOKAHEAD_EVAL"] = str(C), "0"
    x = torch.from_numpy(wav)[None]
    with torch.inference_mode():
        feats, fl = model.preprocessor(x, torch.tensor([x.shape[1]]))
        with torch.autocast("cpu", torch.bfloat16, enabled=bf16):
            enc, _ = model.encoder(feats, fl)
            return model.head(enc)[0].float()


def read(path):
    import soundfile as sf
    w, sr = sf.read(path, dtype="float32")
    return w if w.ndim == 1 else w.mean(1)


def init_pool():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    load_models()


def work(row):
    w = read(row["path"])
    out = {"id": row["id"]}
    for bf in (False, True):
        k = "bf16" if bf else "fp32"
        out[f"c8_{k}"] = greedy(lp_masked(M["c"], w, 8, bf))
        out[f"c16_{k}"] = greedy(lp_masked(M["c"], w, 16, bf))
        out[f"full_{k}"] = greedy(lp_masked(M["f"], w, 0, bf))
    return out


if __name__ == "__main__":
    from cache_engine import CacheEngine
    load_models()
    rows = {s: [json.loads(l) for l in open(STT / f"eval/manifests/{s}.jsonl", encoding="utf-8")] for s in SETS}
    for s in SETS:
        random.Random(5).shuffle(rows[s])
    lat_utts = [read(r["path"]) for r in rows["dev_fleurs_kk"][:3] + rows["dev_ksc2_spont"][:3]]
    long_audio = np.concatenate(lat_utts)
    res = {"cpu": platform.processor() or "unknown", "latency": {}, "parity_engine_text": {}}
    for T in (1, 4):
        torch.set_num_threads(T)
        for rnd in range(2):
            for bf in (False, True):
                k = f"t{T}_{'bf16' if bf else 'fp32'}"
                for C in (8, 16):
                    ts, texts = [], []
                    for w in lat_utts:
                        eng, lps = CacheEngine(M["c"], C, cpu_bf16=bf), []
                        for i in range(0, len(w), 1600):
                            t0 = time.perf_counter(); new = eng.feed(w[i:i + 1600]); dt = time.perf_counter() - t0
                            if new:
                                ts += [dt / len(new)] * len(new); lps += new
                        lps += eng.finish(); texts.append(greedy(torch.cat(lps)))
                    if rnd == 1:
                        res["latency"].setdefault(k, {})[f"chunk{C * 40}ms_ms"] = {"median": round(1000 * float(np.median(ts)), 1),
                                                                                "p90": round(1000 * float(np.percentile(ts, 90)), 1)}
                        res["parity_engine_text"].setdefault(f"c{C}_t{T}", {})["bf16" if bf else "fp32"] = texts
                for sec in (5, 10, 20):
                    seg = long_audio[:sec * 16000]
                    ts = []
                    for _ in range(3):
                        t0 = time.perf_counter(); lp_masked(M["f"], seg, 0, bf); ts.append(time.perf_counter() - t0)
                    if rnd == 1:
                        res["latency"][k][f"final_{sec}s_ms"] = round(1000 * float(np.median(ts)), 1)
            print(T, rnd, json.dumps({k: v for k, v in res["latency"].items() if k.startswith(f"t{T}")}), flush=True)
    for key, d in res["parity_engine_text"].items():
        res["parity_engine_text"][key] = f"{sum(a == b for a, b in zip(d['fp32'], d['bf16']))}/{len(d['fp32'])} identical"
    import jiwer
    from textnorm import normalize
    sub = [r for s in SETS for r in rows[s][:100]]
    with ProcessPoolExecutor(int(os.environ.get("PROCS", "32")), initializer=init_pool) as ex:
        out = {o["id"]: o for o in ex.map(work, sub, chunksize=2)}
    res["wer"] = {}
    for s in SETS:
        sel = [r for r in rows[s][:100] if normalize(r["ref"])]
        refs = [normalize(r["ref"]) for r in sel]
        res["wer"][s] = {k: round(100 * jiwer.wer(refs, [normalize(out[r["id"]][k]) for r in sel]), 2)
                         for k in ("c8_fp32", "c8_bf16", "c16_fp32", "c16_bf16", "full_fp32", "full_bf16")}
    json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "parity_engine_text"}, indent=1))
    print(json.dumps(res["parity_engine_text"]))
