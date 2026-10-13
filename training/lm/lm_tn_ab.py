"""Experiment A/B (result: neutral, so the release keeps the plain kk LM): kk LM from TN-verbalized text (lm/kk_tn.bin) vs the current kk LM (digit lines dropped), both on
the fine-tuned 600M's cached log-probs. kk_tn gets the same alpha x beta grid on the kk dev sets (selection by mean
no-digit WER, as before); the current kk LM keeps its chosen weights. Reports per set: WER on all utterances (std
norm), no-digit WER, and WER after kk_tn ITN vs RAW refs (itn_score.py's light normalization) - the metric where
spoken-form numbers can count. Writes eval/lm_tn_ab.json and test hyps *.ft-large_ctc-raw+lmkk_tn.beam.jsonl."""
import itertools, json, os, re, sys, unicodedata
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
_ws = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace
# lm_eval reads MODEL and TAG from argv at import: the offline training checkpoint (F_CKPT) and its log-prob cache tag
sys.argv = ["lm_eval", os.environ.get("F_CKPT", str(_ws / "runs/full_multilingual_large_ctc/final_a1.00.ckpt")), "ft-large_ctc-raw"]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "kk_tn")]
import jiwer
import lm_eval as L
from score import load, score
from kk_tn.itn import inverse_normalize


def light(t):  # = itn_score.light (not imported: itn_score runs its whole report at import time)
    t = unicodedata.normalize("NFC", t).lower().replace("ё", "е")
    t = re.sub(r"(?<=\d)[\s  ](?=\d{3}(?!\d))", "", t)
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)
    t = re.sub(r"[‐-―\-_]", " ", t)
    t = re.sub(r"(?<!\d)[.:]|[.:](?!\d)", " ", t)
    t = re.sub(r"[^\w\s.:]", " ", t)
    return " ".join(t.split())

STT = L.STT
DEVS = L.DEV["kk"]
TESTS = ["fleurs_kk", "fleurs_kk_tel", "ksc2_podcasts", "ksc2_crowdsourced"]


def metrics(s, hyps):
    rows = load(STT / f"eval/manifests/{s}.jsonl")
    res, _ = score(rows, [{"id": k, "hyp": v} for k, v in hyps.items()])
    sel = [r for r in rows if r["id"] in hyps and light(r["ref"])]
    itn = 100 * jiwer.wer([light(r["ref"]) for r in sel], [light(inverse_normalize(hyps[r["id"]])) for r in sel])
    return {"all": round(res["all"][1], 2), "no_digit": round(res["no-digit"][1], 2), "itn_raw": round(itn, 2),
            "n_digit_refs": sum(bool(re.search(r"\d", r["ref"])) for r in sel)}


old = tuple(json.load(open(STT / "eval/lm_ft-large_ctc-raw.json"))["kk"])
grid = {}
for a, b in itertools.product(L.ALPHAS, L.BETAS):
    grid[(a, b)] = {s: metrics(s, L.decode(s, "kk_tn", a, b)) for s in DEVS}
    print(f"kk_tn alpha={a} beta={b}: " + " | ".join(f"{s} {m['no_digit']}/{m['itn_raw']}" for s, m in grid[(a, b)].items()), flush=True)
mean = lambda d, k: sum(m[k] for m in d.values()) / len(d)
best = min(grid, key=lambda ab: mean(grid[ab], "no_digit"))
out = {"kk_weights": old, "kk_tn_weights": best, "dev": {}, "test": {}}
for s in DEVS:
    out["dev"][s] = {"greedy": metrics(s, L.greedy(s)), "kk": metrics(s, L.decode(s, "kk", *old)), "kk_tn": grid[best][s]}
for s in TESTS:
    hy = L.decode(s, "kk_tn", *best)
    with open(STT / f"eval/hyps/{s}.ft-large_ctc-raw+lmkk_tn.beam.jsonl", "w", encoding="utf-8") as f:
        for k, v in hy.items():
            f.write(json.dumps({"id": k, "hyp": v}, ensure_ascii=False) + "\n")
    prev = {x["id"]: x["hyp"] for x in map(json.loads, open(STT / f"eval/hyps/{s}.ft-large_ctc-raw+lmkk.beam.jsonl", encoding="utf-8"))}
    out["test"][s] = {"greedy": metrics(s, L.greedy(s)), "kk": metrics(s, prev), "kk_tn": metrics(s, hy)}
    print(s, json.dumps(out["test"][s]), flush=True)
json.dump(out, open(STT / "eval/lm_tn_ab.json", "w"), indent=1, default=list)
for part in ("dev", "test"):
    print(f"\n{part}: set | greedy all/no-digit/ITN | kk LM | kk_tn LM")
    for s, d in out[part].items():
        print(s + " | " + " | ".join(f"{d[k]['all']}/{d[k]['no_digit']}/{d[k]['itn_raw']}" for k in ("greedy", "kk", "kk_tn")))
print("LM_TN_AB_DONE")
