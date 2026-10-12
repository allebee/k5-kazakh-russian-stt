"""Run GigaAM-Multilingual (ctc / large_ctc) on a manifest with its own greedy CTC decoding, on CPU."""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace
_model = None


def init(model_dir):
    global _model
    import torch
    import transformers.dynamic_module_utils as dmu
    from transformers import AutoModel
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    # modeling_gigaam.py imports pyannote lazily, only for transcribe_longform (unused here);
    # skip transformers' static import check instead of installing pyannote.
    dmu.check_imports = lambda filename: []
    _model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).eval()


def work(row):
    import torch
    # Same steps as GigaAM's transcribe(), minus its length guard, so the few FLEURS clips over the
    # longform threshold are decoded in one pass like every other model here.
    m = _model.model
    with torch.inference_mode():
        wav, length = m.prepare_wav(row["path"])
        encoded, encoded_len = m.forward(wav, length)
        text, _ = m._decode(encoded, encoded_len, length, False)[0]
    return {"id": row["id"], "hyp": text}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--model-dir", default=str(ROOT / "models/gigaam-ml-ctc"))
    ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
    with ProcessPoolExecutor(a.procs, initializer=init, initargs=(a.model_dir,)) as ex:
        res = list(ex.map(work, rows, chunksize=4))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(res)} hyps -> {a.out}")
