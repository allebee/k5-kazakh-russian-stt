"""Run the old TorchScript models (asr/kk, asr/ru, asr/rukk) on a manifest.

Greedy CTC exactly as on the model card; also saves per-utterance log-probs (fp16 .npy) so KenLM beam
search (beam_old.py) can run without recomputing the acoustic model.
--norm: 'raw' feeds the waveform as read (model card quick start); 'zmuv' applies wav2vec2's usual
zero-mean/unit-variance input normalization. We don't know which one the model was trained with, so test both.
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: models/old = the v1 model repo
_model = _tokens = None


def load_tokens(path):
    tok = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            s, i = line.rstrip("\n").split("\t")
            tok[int(i)] = s
    return tok


def init(model_name):
    global _model, _tokens
    import torch
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    d = ROOT / "models/old/asr" / model_name
    _model = torch.jit.load(str(d / "model.pt"), map_location="cpu").eval()
    _tokens = load_tokens(d / "tokens.lst")


def work(job):
    import torch
    row, norm, logit_dir = job
    wav, sr = sf.read(row["path"], dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:  # e.g. Golos Ogg-Opus decodes at 48 kHz
        import torchaudio.functional as AF
        wav = AF.resample(torch.from_numpy(np.ascontiguousarray(wav)), sr, 16000).numpy()
    if norm == "zmuv":
        wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
    with torch.inference_mode():
        logits = _model(torch.from_numpy(wav).unsqueeze(0))[0][0]
    lp = torch.log_softmax(logits.float(), -1)
    blank = max(_tokens) + 1
    out, prev = [], None
    for i in lp.argmax(-1).tolist():
        if i != prev and i != blank:
            out.append(_tokens.get(i, ""))
        prev = i
    hyp = " ".join("".join(out).replace("|", " ").replace("_", " ").split())
    if logit_dir:
        np.save(Path(logit_dir) / f"{row['id']}.npy", lp.numpy().astype(np.float16))
    return {"id": row["id"], "hyp": hyp}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["kk", "ru", "rukk"])
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--norm", default="raw", choices=["raw", "zmuv"])
    ap.add_argument("--save-logits", action="store_true")
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
    logit_dir = None
    if a.save_logits:
        logit_dir = Path(a.out).with_suffix(".logits")
        logit_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.procs, initializer=init, initargs=(a.model,)) as ex:
        res = list(ex.map(work, [(r, a.norm, logit_dir) for r in rows], chunksize=4))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(res)} hyps -> {a.out}")
