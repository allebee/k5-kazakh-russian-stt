"""Transcribe an eval manifest with a GigaAM model loaded through the gigaam package: a model name
(e.g. multilingual_ctc) or a fine-tuned Lightning .ckpt from train_utils. Same decoding as run_gigaam.py."""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_model = None


def init(name, device="cpu"):
    global _model
    import torch, gigaam
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    gigaam._CACHE_DIR = os.path.join(os.path.abspath(os.environ.get("K5_STT_ROOT", ".")), ".gigaam-cache")
    if os.environ.get("STREAM_PATCH") == "1":  # chunk-masked eval: CHUNK_EVAL frames (40 ms each), 0 = full
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "service"))
        import streaming_patch
        streaming_patch.enable()
    _model = gigaam.load_model(name, fp16_encoder=False, device=device).eval()


def work(row):
    import torch
    try:
        with torch.inference_mode():
            wav, length = _model.prepare_wav(row["path"])
            enc, enc_len = _model.forward(wav, length)
            text = _model._decode(enc, enc_len, length, False)[0][0]
        return {"id": row["id"], "hyp": text}
    except Exception as e:  # one bad file must not lose the whole set; it scores as all-deletions
        return {"id": row["id"], "hyp": "", "error": f"{type(e).__name__}: {str(e)[:200]}"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
    if a.device == "cuda":  # one process, model loaded once, utterances one by one on the GPU
        init(a.model, "cuda")
        res = [work(r) for r in rows]
    else:
        with ProcessPoolExecutor(a.procs, initializer=init, initargs=(a.model,)) as ex:
            res = list(ex.map(work, rows, chunksize=4))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    errs = [r for r in res if "error" in r]
    print(f"wrote {len(res)} hyps -> {a.out}" + (f" ({len(errs)} FAILED, e.g. {errs[0]['error']})" if errs else ""))
