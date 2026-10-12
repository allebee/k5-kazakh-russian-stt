"""End-to-end test of the STAGED release repo with a clean venv (as a user would run it): load both models through
transformers AutoModel (trust_remote_code), run the two-pass streaming pipeline on simulated calls, report final WER
vs references, partial count, ITN output and hotword path. usage: release_test.py REPO_DIR [n_calls]"""
import json, os, sys, time
from pathlib import Path
import numpy as np, soundfile as sf

root = Path(sys.argv[1]); n_calls = int(sys.argv[2]) if len(sys.argv) > 2 else 5
sys.path.insert(0, str(root / "deploy"))
from twopass import TwoPass, release_paths

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
rows = [json.loads(l) for s in ("dev_fleurs_kk", "dev_fleurs_ru", "dev_ksc2_spont")
        for l in list(open(STT / f"eval/manifests/{s}.jsonl", encoding="utf-8"))[5:5 + n_calls]]
t0 = time.time()
tp = TwoPass(paths=release_paths(root), threads=4, hotwords=["Анкара", "Kaspi"])
print(f"loaded in {time.time() - t0:.0f} s; final model module: {type(tp.fm).__module__}", flush=True)
refs, hyps, n_part = [], [], 0
for r in rows:
    w, sr = sf.read(r["path"], dtype="float32")
    w = np.concatenate([w if w.ndim == 1 else w.mean(1), np.zeros(16000, np.float32)])
    tp.reset(); ev = []
    for i in range(0, len(w), 1600):
        ev += tp.feed(w[i:i + 1600])
    ev += tp.flush()
    fin = [e for e in ev if e["type"] == "final"]; n_part += sum(e["type"] == "partial" for e in ev)
    refs.append(r["ref"]); hyps.append(" ".join(e["text"] for e in fin))
    if len(refs) <= 3:
        print("REF:", r["ref"][:90]); print("ITN:", " ".join(e["text_itn"] for e in fin)[:90])
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import jiwer
from textnorm import normalize
print(f"{len(rows)} utterances, {n_part} partial updates, final WER (kkru LM, before ITN): "
      f"{100 * jiwer.wer([normalize(x) for x in refs], [normalize(x) for x in hyps]):.2f}")
