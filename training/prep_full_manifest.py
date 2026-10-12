"""Training mix for the full run (replay-light): everything new to GigaAM-ML at full weight
(KSD, SOVA, RuLS, FLEURS train, telephony copies, kk+ru concatenations), plus a 30% replay of corpora it already
trained on (KSC2, Golos) to limit forgetting. Writes manifests/train_full.tsv and prints hours per source."""
import os, random
from pathlib import Path

M = Path(os.environ.get("K5_STT_ROOT", ".")).resolve() / "manifests"  # workspace
random.seed(13)
mix = {"kk_ksd": 1, "ru_sova": 1, "ru_ruls": 1, "kk_fleurs": 1, "ru_fleurs": 1, "tel_kk": 1, "tel_ru": 1,
       "cs_concat": 1, "kk_ksc2": 0.3, "ru_golos": 0.3}
out, total = [], 0.0
for name, frac in mix.items():
    rows = [r for r in open(M / f"{name}.tsv", encoding="utf-8").read().splitlines()[1:] if random.random() < frac]
    h = sum(float(r.split("\t")[1]) for r in rows) / 3600
    total += h
    print(f"{name:10} x{frac:<4} {len(rows):8d} utts {h:8.1f} h", flush=True)
    out += rows
random.shuffle(out)
(M / "train_full.tsv").write_text("path\tduration\ttranscription\n" + "\n".join(out) + "\n", encoding="utf-8")
print(f"train_full.tsv: {len(out)} utts, {total:.1f} h")
