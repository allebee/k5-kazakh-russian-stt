"""Self-test of the deployable two-pass system on simulated calls (3 dev utterances joined with ~1 s pauses, the same
construction as the engine sims). Reports final WER (greedy-equivalent text and LM text, before ITN), ITN examples,
compute per partial update and per final, and checks that finals are never revised (no audio decoded twice).
usage: selftest.py [--calls 20] [--device cpu] [--chunk-ms 320] [--lm kkru] [--root MODEL_DIR] [--manifests M.jsonl ...]
Manifests are JSONL rows {"path": wav, "ref": text}; default: the three dev sets built by eval/ under $K5_STT_ROOT."""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np

from twopass import TwoPass, SR, release_paths

ap = argparse.ArgumentParser()
ap.add_argument("--calls", type=int, default=20)
ap.add_argument("--device", default="cpu")
ap.add_argument("--chunk-ms", type=int, default=320)
ap.add_argument("--lm", default="kkru")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--out", default="selftest_result.json")
ap.add_argument("--root", default=os.environ.get("K5_STT_MODEL_DIR"),
                help="downloaded k5-kazakh-russian-stt model repo (default: $K5_STT_MODEL_DIR; else the training-side layout)")
ap.add_argument("--manifests", nargs="+", help="eval manifests (default: dev_fleurs_kk, dev_fleurs_ru, dev_ksc2_spont)")
a = ap.parse_args()

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))  # textnorm
import jiwer, soundfile as sf
from textnorm import normalize

man = a.manifests or [STT / f"eval/manifests/{s}.jsonl" for s in ("dev_fleurs_kk", "dev_fleurs_ru", "dev_ksc2_spont")]
rows = [json.loads(l) for m in man for l in open(m, encoding="utf-8")]
random.Random(1).shuffle(rows)
tp = TwoPass(paths=release_paths(a.root) if a.root else None, device=a.device, chunk_ms=a.chunk_ms, lm=a.lm, threads=a.threads)
refs, hyps, part_ms, fin_ms, lm_ms, itn_ex, overlap = [], [], [], [], [], [], 0
for i in range(a.calls):
    grp = rows[i * 3:(i + 1) * 3]
    wav = []
    for r in grp:
        w, sr = sf.read(r["path"], dtype="float32")
        wav += [w if w.ndim == 1 else w.mean(1), np.zeros(int(SR * random.Random(i).uniform(0.9, 1.3)), np.float32)]
    wav = np.concatenate(wav)
    tp.reset()
    evs = []
    for j in range(0, len(wav), SR // 10):
        evs += tp.feed(wav[j:j + SR // 10])
    evs += tp.flush()
    finals = [e for e in evs if e["type"] == "final"]
    overlap += sum(1 for x, y in zip(finals, finals[1:]) if y["start"] < x["end"] - 1e-6)
    part_ms += [e["compute_ms"] for e in evs if e["type"] == "partial"]
    fin_ms += [e["am_ms"] for e in finals]; lm_ms += [e["lm_ms"] for e in finals]
    refs.append(" ".join(normalize(r["ref"]) for r in grp))
    hyps.append(normalize(" ".join(e["text"] for e in finals)))
    itn_ex += [(e["text"], e["text_itn"]) for e in finals if e["text"] != e["text_itn"]][:1]
res = {"calls": a.calls, "chunk_ms": a.chunk_ms, "device": a.device, "lm": a.lm,
       "final_wer_lm_before_itn": round(100 * jiwer.wer(refs, hyps), 2),
       "partial_update_compute_ms": {"mean": round(float(np.mean(part_ms)), 1), "p90": round(float(np.percentile(part_ms, 90)), 1)},
       "final_am_ms": {"mean": round(float(np.mean(fin_ms)), 1), "p90": round(float(np.percentile(fin_ms, 90)), 1)},
       "final_lm_beam_ms": {"mean": round(float(np.mean(lm_ms)), 1), "p90": round(float(np.percentile(lm_ms, 90)), 1)},
       "overlapping_finals": overlap, "itn_examples": itn_ex[:5]}
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
