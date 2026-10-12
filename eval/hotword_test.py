"""Contextual biasing (pyctcdecode hotwords) on the final pass: recall of rare words with vs without biasing.
Targets: words in dev references that are NOT in the kkru LM unigram list (>= 5 letters, mostly names and rare terms),
in utterances where they occur. Hotword list = all targets + DISTRACT rare words from TEST references (never spoken in
dev), so the list is realistic (most entries absent). Decoding reuses the cached FT log-probs
(eval/logits/ft-large_ctc-raw/<dev set>.npz), kkru LM at its dev-tuned weights.
Reports per hotword weight: target recall (word present in the hypothesis), WER on target utterances, and WER on
utterances WITHOUT targets (false-alarm cost), plus greedy for reference.
usage: hotword_test.py OUT.json   env: DISTRACT=200, WEIGHTS=0,5,10,20, PROCS=32"""
import json, os, random, re, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "service")]
from textnorm import normalize
SETS = ["dev_fleurs_kk", "dev_fleurs_ru", "dev_ksc2_spont"]
D = {}


def init(hot, weight):
    from pyctcdecode import build_ctcdecoder
    vocab = json.load(open(STT / "models/gigaam-ml-ctc/config.json"))["cfg"]["model"]["cfg"]["decoding"]["vocabulary"]
    a, b = json.load(open(STT / "eval/lm_ft-large_ctc-raw.json"))["kkru"]
    D["dec"] = build_ctcdecoder(vocab + [""], kenlm_model_path=str(STT / "lm/kkru.bin"),
                                unigrams=(STT / "lm/kkru.unigrams.txt").read_text(encoding="utf-8").split(), alpha=a, beta=b)
    D["hot"], D["w"] = hot, weight


def dec(item):
    k, lp = item
    if D["hot"] and D["w"] > 0:
        return k, D["dec"].decode(lp.astype(np.float32), beam_width=64, hotwords=D["hot"], hotword_weight=D["w"])
    return k, D["dec"].decode(lp.astype(np.float32), beam_width=64)


if __name__ == "__main__":
    import jiwer
    uni = set((STT / "lm/kkru.unigrams.txt").read_text(encoding="utf-8").split())
    rows, logits = [], {}
    for s in SETS:
        rows += [dict(json.loads(l), set=s) for l in open(STT / f"eval/manifests/{s}.jsonl", encoding="utf-8")]
        z = np.load(STT / f"eval/logits/ft-large_ctc-raw/{s}.npz")
        logits.update({k: z[k] for k in z.files})
    rows = [r for r in rows if r["id"] in logits and normalize(r["ref"])]
    rare = lambda w: len(w) >= 5 and w not in uni and not re.search(r"\d", w)
    tgt = {r["id"]: sorted({w for w in normalize(r["ref"]).split() if rare(w)}) for r in rows}
    targets = sorted({w for ws in tgt.values() for w in ws})
    test_words = set()
    for s in ("fleurs_kk", "fleurs_ru", "ksc2_podcasts", "cs_ksc2"):
        for l in open(STT / f"eval/manifests/{s}.jsonl", encoding="utf-8"):
            test_words |= {w for w in normalize(json.loads(l)["ref"]).split() if rare(w)}
    distract = sorted(test_words - set(targets))
    random.Random(1).shuffle(distract)
    hot = targets + distract[:int(os.environ.get("DISTRACT", "200"))]
    with_t = [r for r in rows if tgt[r["id"]]]; without = [r for r in rows if not tgt[r["id"]]]
    print(f"{len(targets)} target words in {len(with_t)} of {len(rows)} dev utterances; hotword list {len(hot)}", flush=True)
    res = {"n_targets": len(targets), "n_target_utts": len(with_t), "n_other_utts": len(without), "hotword_list": len(hot),
           "examples": targets[:25], "runs": {}}
    items = [(r["id"], logits[r["id"]]) for r in rows]
    for w in [float(x) for x in os.environ.get("WEIGHTS", "0,5,10,20").split(",")]:
        with ProcessPoolExecutor(int(os.environ.get("PROCS", "32")), initializer=init, initargs=(hot, w)) as ex:
            hy = dict(ex.map(dec, items, chunksize=8))
        hit = sum(t in normalize(hy[r["id"]]).split() for r in with_t for t in tgt[r["id"]])
        tot = sum(len(tgt[r["id"]]) for r in with_t)
        wer = lambda rs: round(100 * jiwer.wer([normalize(r["ref"]) for r in rs], [normalize(hy[r["id"]]) for r in rs]), 2)
        # false alarms: distractor hotwords appearing in hypotheses of utterances whose reference lacks them
        fa = sum(1 for r in rows for d in distract[:int(os.environ.get("DISTRACT", "200"))]
                 if d in normalize(hy[r["id"]]).split() and d not in normalize(r["ref"]).split())
        res["runs"][f"weight_{w:g}"] = {"target_recall": round(hit / max(1, tot), 4), "wer_target_utts": wer(with_t),
                                        "wer_other_utts": wer(without), "wer_all": wer(rows), "distractor_false_alarms": fa}
        print(f"weight {w:g}: {res['runs'][f'weight_{w:g}']}", flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
