"""Stage C v2 mix (the longer streaming variant, NOT released: no better than C v1 at R = 0): like the offline training
mix but with a higher Russian share (2026-10-08): Golos replay 60% (was 30%), SOVA and RuLS x1.5 (all rows + a random
half again). Reads the per-source manifests written by build_manifests.py / make_augment.py in $K5_STT_ROOT/manifests;
writes manifests/train_c2.tsv and prints hours per language."""
import os, random
from pathlib import Path

M = Path(os.environ.get("K5_STT_ROOT", ".")).resolve() / "manifests"
random.seed(17)
mix = {"kk_ksd": 1, "kk_fleurs": 1, "tel_kk": 1, "cs_concat": 1, "kk_ksc2": 0.3,
       "ru_fleurs": 1, "tel_ru": 1, "ru_golos": 0.6, "ru_sova": 1.5, "ru_ruls": 1.5}
out, hours = [], {"kk": 0.0, "ru": 0.0, "cs": 0.0}
for name, frac in mix.items():
    rows = open(M / f"{name}.tsv", encoding="utf-8").read().splitlines()[1:]
    take = [r for r in rows if random.random() < min(frac, 1)] + ([r for r in rows if random.random() < frac - 1] if frac > 1 else [])
    h = sum(float(r.split("\t")[1]) for r in take) / 3600
    hours["cs" if name == "cs_concat" else ("kk" if "kk" in name else "ru")] += h
    print(f"{name:10} x{frac:<4} {len(take):8d} utts {h:8.1f} h", flush=True)
    out += take
random.shuffle(out)
(M / "train_c2.tsv").write_text("path\tduration\ttranscription\n" + "\n".join(out) + "\n", encoding="utf-8")
tot = sum(hours.values())
print(f"train_c2.tsv: {len(out)} utts, {tot:.1f} h | " + " ".join(f"{k} {100 * v / tot:.0f}%" for k, v in hours.items()))
