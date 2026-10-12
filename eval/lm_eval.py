"""KenLM beam search (pyctcdecode) on top of a GigaAM CTC model, with LM weights tuned on DEV sets only.

1. Caches per-utterance CTC log-probs for every set (CPU pool) in eval/logits/<tag>/<set>.npz.
2. Grid-searches alpha x beta per LM on its dev sets (mean no-digit WER) and writes the choice to eval/lm_<tag>.json.
3. Decodes every test set with its LM at the chosen weights -> eval/hyps/<set>.<tag>+lm<lm>.beam.jsonl
usage: lm_eval.py MODEL TAG   (MODEL: gigaam name or .ckpt; TAG: e.g. gigaam-ml-large-ctc)"""
import itertools, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
sys.path.insert(0, str(Path(__file__).resolve().parent))
MODEL, TAG = sys.argv[1], sys.argv[2]
DEV = {"kk": ["dev_fleurs_kk", "dev_fleurs_kk_tel", "dev_ksc2_spont"], "ru": ["dev_fleurs_ru", "dev_fleurs_ru_tel"],
       "kkru": ["dev_fleurs_kk", "dev_fleurs_ru", "dev_ksc2_spont"]}
TEST = {"fleurs_kk": "kk", "fleurs_kk_tel": "kk", "fleurs_ru": "ru", "fleurs_ru_tel": "ru", "ksc2_podcasts": "kk",
        "golos_farfield": "ru", "cs_tim2190": "kkru", "cs_ksc2": "kkru"}
ALPHAS = [float(x) for x in os.environ.get("ALPHAS", "0.1,0.2,0.3,0.5").split(",")]
BETAS = [float(x) for x in os.environ.get("BETAS", "-1.0,0.0,1.0").split(",")]
_m = _dec = None


def init_model():
    global _m
    import torch, gigaam
    torch.set_num_threads(4)
    gigaam._CACHE_DIR = str(STT / ".gigaam-cache")
    _m = gigaam.load_model(MODEL, fp16_encoder=False, device="cpu").eval()


def logp(row):
    import torch
    try:
        with torch.inference_mode():
            wav, length = _m.prepare_wav(row["path"])
            enc, enc_len = _m.forward(wav, length)
            return row["id"], _m.head(enc)[0].float().numpy().astype(np.float16)
    except Exception:
        return row["id"], None


def cache(s):
    out = STT / f"eval/logits/{TAG}/{s}.npz"
    if not out.exists():
        rows = [json.loads(l) for l in open(STT / f"eval/manifests/{s}.jsonl", encoding="utf-8")]
        with ProcessPoolExecutor(int(os.environ.get("LM_PROCS", "12")), initializer=init_model) as ex:
            res = dict(ex.map(logp, rows, chunksize=4))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **{k: v for k, v in res.items() if v is not None})
    return out


def init_dec(lm, alpha, beta):
    global _dec
    from pyctcdecode import build_ctcdecoder
    vocab = json.load(open(STT / "models/gigaam-ml-ctc/config.json"))["cfg"]["model"]["cfg"]["decoding"]["vocabulary"]
    uni = STT / f"lm/{lm}.unigrams.txt"  # words seen >=3 times in the LM text (binary KenLM files don't list them)
    unigrams = uni.read_text(encoding="utf-8").split() if uni.exists() else None
    _dec = build_ctcdecoder(vocab + [""], kenlm_model_path=str(STT / f"lm/{lm}.bin"), unigrams=unigrams,
                            alpha=alpha, beta=beta)


def dec(item):
    k, lp = item
    return k, _dec.decode(lp.astype(np.float32), beam_width=64)


def decode(s, lm, alpha, beta):
    z = np.load(cache(s))
    with ProcessPoolExecutor(int(os.environ.get("DEC_PROCS", "24")), initializer=init_dec, initargs=(lm, alpha, beta)) as ex:
        return dict(ex.map(dec, [(k, z[k]) for k in z.files], chunksize=8))


def greedy(s):
    vocab = json.load(open(STT / "models/gigaam-ml-ctc/config.json"))["cfg"]["model"]["cfg"]["decoding"]["vocabulary"]
    z, out = np.load(cache(s)), {}
    for k in z.files:
        ids, chars, prev = z[k].argmax(-1), [], -1
        for i in ids:
            if i != prev and i != len(vocab):
                chars.append(vocab[i])
            prev = i
        out[k] = " ".join("".join(chars).split())
    return out


def wer_nodigit(s, hyps):
    from score import load, score
    res, _ = score(load(STT / f"eval/manifests/{s}.jsonl"), [{"id": k, "hyp": v} for k, v in hyps.items()])
    return res["no-digit"][1]


if __name__ == "__main__":
    choice = {}
    for lm, devs in DEV.items():
        grid = {}
        for a, b in itertools.product(ALPHAS, BETAS):
            grid[(a, b)] = sum(wer_nodigit(s, decode(s, lm, a, b)) for s in devs) / len(devs)
            print(f"{lm} alpha={a} beta={b}: dev no-digit WER {grid[(a, b)]:.2f}", flush=True)
        g = sum(wer_nodigit(s, greedy(s)) for s in devs) / len(devs)
        best = min(grid, key=grid.get)
        choice[lm] = best if grid[best] < g else None  # the LM must beat greedy on dev, else it's off
        print(f"{lm}: greedy dev {g:.2f} | best beam {best} {grid[best]:.2f} -> "
              f"{'use LM' if choice[lm] else 'LM OFF (greedy wins)'}", flush=True)
    json.dump({k: (list(v) if v else None) for k, v in choice.items()}, open(STT / f"eval/lm_{TAG}.json", "w"))
    for s, lm in TEST.items():
        if not (STT / f"eval/manifests/{s}.jsonl").exists():
            continue
        if choice[lm] is None:
            print(f"{s}: LM {lm} off (greedy wins on dev)", flush=True)
            continue
        hyps = decode(s, lm, *choice[lm])
        with open(STT / f"eval/hyps/{s}.{TAG}+lm{lm}.beam.jsonl", "w", encoding="utf-8") as f:
            for k, v in hyps.items():
                f.write(json.dumps({"id": k, "hyp": v}, ensure_ascii=False) + "\n")
        print(f"{s}: decoded with lm {lm} {choice[lm]}", flush=True)
    print("LM_EVAL_DONE")
