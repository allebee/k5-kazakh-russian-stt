"""KenLM beam search over log-probs saved by run_old.py --save-logits, with the model card's
production settings (lm/<lm>/ lexicon + 4-gram, beam 1000, lm_weight 1.2, word_score 1, sil_score -0.2)."""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: models/old = the v1 model repo
_dec = None


def init(model, lm, beam):
    global _dec
    import torch
    from torchaudio.models.decoder import ctc_decoder
    torch.set_num_threads(1)
    tok = {}
    for line in open(ROOT / f"models/old/asr/{model}/tokens.lst", encoding="utf-8"):
        if line.strip():
            s, i = line.rstrip("\n").split("\t")
            tok[int(i)] = s
    tokens = [tok[i] for i in range(max(tok) + 1)] + ["<blank>"]
    _dec = ctc_decoder(
        lexicon=str(ROOT / f"models/old/lm/{lm}/words.lst"), tokens=tokens,
        lm=str(ROOT / f"models/old/lm/{lm}/lm.bin"), nbest=1,
        beam_size=beam, beam_size_token=beam, beam_threshold=15,
        lm_weight=1.2, word_score=1.0, sil_score=-0.2,
        blank_token="<blank>", sil_token="|", unk_word="[UNK]")


def work(path):
    import torch
    lp = torch.from_numpy(np.load(path).astype(np.float32)).unsqueeze(0)
    r = _dec(lp)
    return {"id": Path(path).stem, "hyp": " ".join(r[0][0].words) if r[0] else ""}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["kk", "ru", "rukk"])
    ap.add_argument("logits_dir")
    ap.add_argument("out")
    ap.add_argument("--lm", help="lm folder (default: same as model)")
    ap.add_argument("--beam", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    files = sorted(str(p) for p in Path(a.logits_dir).glob("*.npy"))
    with ProcessPoolExecutor(a.procs, initializer=init, initargs=(a.model, a.lm or a.model, a.beam)) as ex:
        res = list(ex.map(work, files, chunksize=2))
    with open(a.out, "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(res)} hyps -> {a.out}")
