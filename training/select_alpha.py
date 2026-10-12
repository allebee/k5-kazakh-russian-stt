"""Pick the checkpoint with the lowest mean no-digit WER over the dev sets. Prints the winning tag and writes JSON.
usage: select_alpha.py OUT.json TAG [TAG ...]   (hyps: eval/hyps/<devset>.<TAG>.greedy.jsonl)"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from score import load, score

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
DEV = ["dev_fleurs_kk", "dev_fleurs_ru", "dev_fleurs_kk_tel", "dev_fleurs_ru_tel", "dev_ksc2_spont"]
out, tags = sys.argv[1], sys.argv[2:]
table = {}
for tag in tags:
    w = []
    for d in DEV:
        h = STT / f"eval/hyps/{d}.{tag}.greedy.jsonl"
        if not h.exists():
            sys.exit(f"missing {h}")
        s, missing = score(load(STT / f"eval/manifests/{d}.jsonl"), load(h))
        if missing:
            sys.exit(f"incomplete {h}: {len(missing)} missing")
        w.append(s["no-digit"][1])
    table[tag] = {"mean_nodigit_wer": sum(w) / len(w), **dict(zip(DEV, w))}
best = min(table, key=lambda t: table[t]["mean_nodigit_wer"])
json.dump({"best": best, "table": table}, open(out, "w"), indent=1)
for t, v in table.items():
    print(f"{t:40} mean {v['mean_nodigit_wer']:.2f}" + ("   <- best" if t == best else ""))
print(best)
