"""Transcribe several eval manifests with ONE model load (the per-set reload cost ~2 min each on the 600M).
usage: eval_many.py CKPT_OR_NAME DEVICE MANIFEST=OUT [MANIFEST=OUT ...]
Skips outputs that already have one line per manifest row. Writes via a temp file + atomic rename, so a concurrent
reader never sees a partial file. Same decoding as run_gigaam_ckpt.py (incl. STREAM_PATCH / CHUNK_EVAL)."""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_gigaam_ckpt as R


def n_lines(p):
    with open(p, encoding="utf-8") as f:
        return sum(1 for _ in f)


if __name__ == "__main__":
    ck, device, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
    todo = []
    for pr in pairs:
        man, out = pr.split("=", 1)
        rows = [json.loads(l) for l in open(man, encoding="utf-8")]
        if not (Path(out).exists() and n_lines(out) == len(rows)):
            todo.append((rows, out))
    if not todo:
        sys.exit(0)
    R.init(ck, device)
    for rows, out in todo:
        res = [R.work(r) for r in rows]
        tmp = f"{out}.tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, out)
        errs = sum("error" in r for r in res)
        print(f"wrote {len(res)} hyps -> {out}" + (f" ({errs} FAILED)" if errs else ""), flush=True)
