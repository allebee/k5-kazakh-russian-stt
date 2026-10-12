"""Dev sets for model selection (WiSE-FT alpha, streaming checkpoints), so test sets stay untouched:
dev_fleurs_kk / dev_fleurs_ru (FLEURS dev), their telephony-sim copies (same chain as the test copies), and
dev_ksc2_spont (KSC2 Dev tv_news/radio/podcasts/talkshow, at most 600, minus the code-switch test rows)."""
import json, os, random, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import prep_eval as P

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
KSC2 = Path(os.environ.get("KSC2_DIR", STT / "data/ksc2/extracted/ISSAI_KSC2")) / "Dev"
random.seed(3)
with ProcessPoolExecutor(64) as pool:
    for lang, short in (("kk_kz", "kk"), ("ru_ru", "ru")):
        base, rows, seen = STT / f"data/fleurs/extracted/{lang}/dev", [], set()
        for line in open(STT / f"data/fleurs/data/{lang}/dev.tsv", encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if p[1] not in seen and (base / p[1]).exists():
                seen.add(p[1]); rows.append({"id": p[1], "path": str(base / p[1]), "ref": p[2], "lang": short})
        P.write_manifest(f"dev_fleurs_{short}", rows)
        tel = STT / f"data/eval/dev_fleurs_{short}_tel"; tel.mkdir(parents=True, exist_ok=True)
        codecs = list(pool.map(P.telephony, [(Path(r["path"]), tel / r["id"]) for r in rows], chunksize=8))
        P.write_manifest(f"dev_fleurs_{short}_tel", [dict(r, path=str(tel / r["id"]), codec=c) for r, c in zip(rows, codecs)])
held = {json.loads(l)["path"] for l in open(STT / "eval/manifests/cs_ksc2.jsonl", encoding="utf-8")}
rows = []
for src in ("tv_news", "radio", "podcasts", "talkshow"):
    for txt in sorted((KSC2 / src).glob("*.txt")):
        audio = txt.parent / (txt.name.split(".")[0] + ".flac")
        if audio.exists() and str(audio) not in held:
            rows.append({"id": f"ksc2_Dev_{src}_{audio.stem}", "path": str(audio), "ref": txt.read_text(encoding="utf-8").strip(), "lang": "kk"})
P.write_manifest("dev_ksc2_spont", random.sample(rows, min(600, len(rows))))
print("DEV_SETS_DONE")
