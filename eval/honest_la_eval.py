"""Honest vs leaky lookahead for a stage-C checkpoint (CPU).
- leaky: streaming_patch single-pass mask with LOOKAHEAD_EVAL=R (what the chain's *-la8 evals measured)
- honest: cache_engine with right context R (each chunk computed with its next R frames only; streamable)
- R=0 via the cache engine (= masked R=0 exactly)
Plus a direct leak test: logits of the frames up to chunk k from the full utterance vs from audio cut right after
chunk k's lookahead (+40 ms front-end margin). Any difference = the output depended on audio it could not have seen.
usage: honest_la_eval.py CKPT OUT.json  (env: N=100 per set, PROCS=32, SETS=...)"""
import json, os, random, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "service")]
CK, OUT = sys.argv[1], sys.argv[2]
SETS = os.environ.get("SETS", "dev_fleurs_kk,dev_fleurs_ru,dev_ksc2_spont").split(",")
CONFIGS = [(4, 8), (8, 8), (16, 8), (8, 0), (16, 0)]
M = {}


def init():
    import torch, gigaam, streaming_patch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    gigaam._CACHE_DIR = str(STT / ".gigaam-cache")
    streaming_patch.enable()
    M["m"] = gigaam.load_model(CK, fp16_encoder=False, device="cpu").eval()
    M["vocab"] = [str(v) for v in M["m"].decoding.tokenizer.vocab]


def greedy(lp):
    out, prev = [], -1
    for i in lp.argmax(-1).tolist():
        if i != prev and i != len(M["vocab"]):
            out.append(M["vocab"][i])
        prev = i
    return " ".join("".join(out).split())


def masked(wav, C, R):
    import torch
    os.environ["CHUNK_EVAL"], os.environ["LOOKAHEAD_EVAL"] = str(C), str(R)
    x = torch.from_numpy(wav)[None]
    with torch.inference_mode():
        enc, _ = M["m"].forward(x, torch.tensor([x.shape[1]]))
        return M["m"].head(enc)[0]


def work(job):
    import torch, soundfile as sf
    from cache_engine import CacheEngine
    row, leak_test = job
    wav, sr = sf.read(row["path"], dtype="float32")
    wav = wav if wav.ndim == 1 else wav.mean(1)
    out = {"id": row["id"]}
    for C, R in CONFIGS:
        eng = CacheEngine(M["m"], C, lookahead=R)
        lps = []
        for i in range(0, len(wav), 1600):
            lps += eng.feed(wav[i:i + 1600])
        lps += eng.finish()
        out[f"honest_c{C}_r{R}"] = greedy(torch.cat(lps))
        if R:
            out[f"leaky_c{C}_r{R}"] = greedy(masked(wav, C, R))
    if leak_test:
        for C, R in ((16, 8), (8, 8), (16, 0)):
            full = masked(wav, C, R)
            k = max(1, full.shape[0] // C // 2)  # chunks 0..k-1 are compared (k = a chunk in the middle)
            cut = (k * C + R) * 640 + 640        # samples for frames < k*C + R, plus the 40 ms front-end lookahead
            n = k * C
            # leaky single-pass mask: does chunk < k depend on audio after chunk_end + R?
            out[f"leak_masked_c{C}_r{R}"] = float((full[:n] - masked(wav[:cut], C, R)[:n]).abs().max())
            # honest engine: same question; feed the cut audio without finish() -> must give the same first k chunks
            e_full, e_cut = CacheEngine(M["m"], C, lookahead=R), CacheEngine(M["m"], C, lookahead=R)
            a = torch.cat(e_full.feed(wav) + e_full.finish())[:n]
            b = torch.cat(e_cut.feed(wav[:cut]))[:n]
            out[f"leak_honest_c{C}_r{R}"] = float((a - b).abs().max()) if b.shape[0] == n else float("nan")
    return out


if __name__ == "__main__":
    import jiwer
    from textnorm import normalize
    rows = {}
    for s in SETS:
        r = [json.loads(l) for l in open(STT / f"eval/manifests/{s}.jsonl", encoding="utf-8")]
        random.Random(3).shuffle(r)
        rows[s] = r[:int(os.environ.get("N", "100"))]
    jobs = [(r, i < 3) for s in SETS for i, r in enumerate(rows[s])]
    with ProcessPoolExecutor(int(os.environ.get("PROCS", "32")), initializer=init) as ex:
        res = {o["id"]: o for o in ex.map(work, jobs)}
    keys = [f"honest_c{C}_r{R}" for C, R in CONFIGS] + [f"leaky_c{C}_r{R}" for C, R in CONFIGS if R]
    table = {}
    for s in SETS:
        sel = [r for r in rows[s] if normalize(r["ref"])]
        refs = [normalize(r["ref"]) for r in sel]
        table[s] = {k: round(100 * jiwer.wer(refs, [normalize(res[r["id"]][k]) for r in sel]), 2) for k in keys}
    leak = {f"{kind}_c{C}_r{R}": [round(o[f"leak_{kind}_c{C}_r{R}"], 5) for o in res.values() if f"leak_{kind}_c{C}_r{R}" in o]
            for kind in ("masked", "honest") for C, R in ((16, 8), (8, 8), (16, 0))}
    json.dump({"ckpt": CK, "wer": table, "leak_test_max_abs_dlogp": leak}, open(OUT, "w"), indent=1)
    print("WER (greedy) | " + " | ".join(keys))
    for s, d in table.items():
        print(s + " | " + " | ".join(str(d[k]) for k in keys))
    print("leak test: max |dlogp| of chunks before k, full audio vs audio cut at chunk_end(k-1) + R + 40 ms "
          "(9 utterances; ~0 = streamable):")
    for k, v in leak.items():
        print(f"  {k}: {v}")
    print("HONEST_LA_DONE")
