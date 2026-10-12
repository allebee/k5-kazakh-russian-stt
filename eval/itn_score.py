"""ITN-aware scoring: hypotheses -> kk_tn.itn.inverse_normalize -> compared with the RAW
references (digits as written), under light normalization only: lowercase, ё->е, digit groups joined ("50 000" ->
"50000"), decimal comma -> dot, "." and ":" kept only inside numbers, other punctuation and hyphens -> space.
Also a regression check on references WITHOUT digits: how many hypotheses ITN changes there, and the WER delta
(any change there is a false conversion, e.g. "бір күні" -> "1 күні").
"v1" = the earlier alibiserikbay/kazakh-russian-mixed-stt (hyps from benchmarks/run_old.py + beam_old.py).
usage: itn_score.py OUT.txt"""
import json, os, re, sys, unicodedata
from pathlib import Path

import jiwer

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kk_tn"))
from kk_tn.itn import inverse_normalize


def light(t):
    t = unicodedata.normalize("NFC", t).lower().replace("ё", "е")
    t = re.sub(r"(?<=\d)[\s  ](?=\d{3}(?!\d))", "", t)
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)
    t = re.sub(r"[‐-―\-_]", " ", t)
    t = re.sub(r"(?<!\d)[.:]|[.:](?!\d)", " ", t)
    t = re.sub(r"[^\w\s.:]", " ", t)
    return " ".join(t.split())


SYSTEMS = {"v1 greedy": None, "v1 +KenLM": None, "GigaAM-600M zero-shot": "gigaam-ml-large-ctc.greedy",
           "ours fine-tuned": "ft-large_ctc-raw.greedy"}
USER = {"fleurs_kk": "kk", "fleurs_kk_tel": "kk", "fleurs_ru": "ru", "fleurs_ru_tel": "ru", "cs_tim2190": "rukk",
        "cs_ksc2": "rukk", "ksc2_crowdsourced": "kk", "ksc2_tv_news": "rukk", "ksc2_talkshow": "rukk",
        "ksc2_radio": "rukk", "ksc2_podcasts": "rukk", "ksc2_parliament": "rukk", "golos_crowd": "ru", "golos_farfield": "ru"}
out = ["set | system | WER std-norm (no ITN) | WER with ITN vs raw refs | no-digit refs: WER without/with ITN, hyps changed",
       "---|---|---|---|---"]
examples = []
for s, m in USER.items():
    man = STT / f"eval/manifests/{s}.jsonl"
    if not man.exists():
        continue
    rows = [json.loads(l) for l in open(man, encoding="utf-8")]
    for name, tag in SYSTEMS.items():
        tag = tag or (f"old-{m}.raw.greedy" if name == "v1 greedy" else f"old-{m}.raw.beam-lm{m}")
        f = STT / f"eval/hyps/{s}.{tag}.jsonl"
        if not f.exists():
            continue
        hy = {x["id"]: x["hyp"] for x in map(json.loads, open(f, encoding="utf-8"))}
        sel = [r for r in rows if r["id"] in hy and light(r["ref"])]
        refs = [light(r["ref"]) for r in sel]
        plain = [light(hy[r["id"]]) for r in sel]
        itn = [light(inverse_normalize(hy[r["id"]])) for r in sel]
        nd = [i for i, r in enumerate(sel) if not re.search(r"\d", r["ref"])]
        changed = [i for i in nd if plain[i] != itn[i]]
        w = lambda R, H: 100 * jiwer.wer(R, H) if R else float("nan")
        out.append(f"{s} | {name} | {w(refs, plain):.2f} | {w(refs, itn):.2f} | "
                   f"{w([refs[i] for i in nd], [plain[i] for i in nd]):.2f} / {w([refs[i] for i in nd], [itn[i] for i in nd]):.2f}, "
                   f"{len(changed)} of {len(nd)} changed")
        if name == "ours fine-tuned":  # show kk_tn's RAW output, not the scorer's normalized string
            for i in changed[:3]:
                raw_h = hy[sel[i]["id"]]
                examples.append(f"{s}: «{raw_h}» -> «{inverse_normalize(raw_h)}» (raw ref: «{sel[i]['ref']}»)")
text = "\n".join(out) + "\n\nFalse-conversion examples (no-digit refs, ours fine-tuned):\n" + "\n".join(examples) + "\n"
Path(sys.argv[1]).write_text(text, encoding="utf-8")
print(text)
