"""Fixed benchmark subsets (<= MAX utterances per set, seeded) so that slow external models and our systems are scored
on exactly the same utterances. Writes eval/manifests/bench_<set>.jsonl."""
import json, os, random
from pathlib import Path
E = Path(os.environ.get("K5_STT_ROOT", ".")).resolve() / "eval/manifests"  # workspace
MAX = int(os.environ.get("MAX", "600"))
for s in ("fleurs_kk", "fleurs_ru", "cs_ksc2", "ksc2_podcasts", "ksc2_talkshow", "ksc2_tv_news", "golos_farfield"):
    rows = [json.loads(l) for l in open(E / f"{s}.jsonl", encoding="utf-8")]
    rows.sort(key=lambda r: r["id"]); random.Random(7).shuffle(rows)
    with open(E / f"bench_{s}.jsonl", "w", encoding="utf-8") as f:
        for r in rows[:MAX]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(s, len(rows), "->", min(MAX, len(rows)))
