"""Open-model benchmark table: every system on the same bench subsets, scored three ways.
  all       : our standard scoring (lowercase, ё->е, punctuation and hyphens -> space, digits kept), every utterance
  no-digit  : same, only utterances whose reference has no digits (fair across digit vs spelled-out writers)
  ITN-fair  : itn_score's light normalization vs the RAW reference; spoken-form systems (ours, v1) go through
              kk_tn ITN first, written-form systems (Whisper family) are scored as they write (digits already).
              Meaningful where references write digits (FLEURS); on KSC2/Golos references spell numbers out.
usage: bench_score.py > eval/BENCH_open_models.md"""
import glob, json, os, re, sys, unicodedata
from pathlib import Path
STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "kk_tn")]
import jiwer
from score import load, score
from kk_tn.itn import inverse_normalize

SETS = {"fleurs_kk200": ("kk", "kk"), "fleurs_kk": ("kk", "kk"), "fleurs_ru": ("ru", "ru"), "cs_ksc2": ("rukk", "kkru"), "ksc2_podcasts": ("rukk", "kk"),
        "ksc2_talkshow": ("rukk", None), "ksc2_tv_news": ("rukk", None), "golos_farfield": ("ru", "ru")}
BENCH = [("shyngys", "shyngys879/kazakh-whisper-large-v3-turbo"), ("shyngys-fb", "shyngys879 + temperature fallback"),
         ("turbo", "openai/whisper-large-v3-turbo (zero-shot)"), ("turbo-fb", "whisper-large-v3-turbo + temperature fallback"),
         ("sybyrla", "InflexionLab/sybyrla"), ("buzz", "BuzzASR/kazakh"), ("omni-ctc1b", "Meta omniASR CTC 1B v2"),
         ("omni-ctc300m", "Meta omniASR CTC 300M v2"), ("omni-ctc3b", "Meta omniASR CTC 3B v2"), ("omni-ctc7b", "Meta omniASR CTC 7B v2 (GPU)"),
         ("omni-llm7b", "Meta omniASR LLM 7B v2, lang-conditioned (GPU)"), ("shyngys-b5gpu", "shyngys879 beam 5 (GPU fp16)"),
         ("turbo-b5gpu", "whisper-large-v3-turbo beam 5 (GPU fp16)"), ("sybyrla-b5gpu", "sybyrla beam 5 (GPU fp16)")]


def light(t):
    t = unicodedata.normalize("NFC", t).lower().replace("ё", "е")
    t = re.sub(r"(?<=\d)[\s  ](?=\d{3}(?!\d))", "", t)
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)
    t = re.sub(r"[‐-―\-_]", " ", t)
    t = re.sub(r"(?<!\d)[.:]|[.:](?!\d)", " ", t)
    t = re.sub(r"[^\w\s.:]", " ", t)
    return " ".join(t.split())


def systems(s):
    um, lm = SETS[s]
    out = [("kazakh-russian-mixed-stt (v1) greedy", f"old-{um}.raw.greedy", True), ("kazakh-russian-mixed-stt (v1) + its KenLM", f"old-{um}.raw.beam-lm{um}", True),
           ("**ours: FT (greedy)**", "ft-large_ctc-raw.greedy", True)]
    if lm:
        out.append(("**ours: FT + KenLM**", f"ft-large_ctc-raw+lm{lm}.beam", True))
    out.append(("ours: streaming partials 640 ms (C v2, R=0)", "stream2-large_ctc-c16-la0.greedy", True))
    out += [(name, f"bench-{tag}", False) for tag, name in BENCH]
    return out


print("# Open Kazakh/Russian ASR models vs ours, on our test subsets and scorer (2026-10-08)\n")
print("fleurs_kk200 = the first 200 utterances of the fleurs_kk bench subset (the slow large-v3 models on CPU). ")
print("WER %. Same utterances for every system (bench subsets: <= 600 per set, seeded). External models: CPU fp32, "
      "greedy, language forced (kk for Kazakh and mixed sets, ru for Russian). ITN-fair: spoken-form systems + kk_tn ITN, "
      "written-form systems as written; vs raw references (meaningful on FLEURS). '–' = not run / not applicable.\n")
for s in SETS:
    rows = load(STT / f"eval/manifests/bench_{s}.jsonl")
    ids = {r["id"] for r in rows}
    print(f"\n## {s} ({len(rows)} utterances, {sum(bool(re.search(r'[0-9]', r['ref'])) for r in rows)} with digits in the reference)\n")
    print("| system | all | no-digit | ITN-fair |\n|---|---|---|---|")
    for name, tag, spoken in systems(s):
        f = STT / f"eval/hyps/{s}.{tag}.jsonl"
        if not f.exists() and s.endswith("200"):  # 200-utterance subset: reuse the full-subset hyps
            f = STT / f"eval/hyps/{s[:-3]}.{tag}.jsonl"
        if not f.exists():
            continue
        hy = {x["id"]: x["hyp"] for x in load(f) if x["id"] in ids}
        if len(hy) < len(ids):
            print(f"| {name} | (missing {len(ids) - len(hy)} hyps) | | |"); continue
        res, _ = score(rows, [{"id": k, "hyp": v} for k, v in hy.items()])
        sel = [r for r in rows if light(r["ref"])]
        fair = 100 * jiwer.wer([light(r["ref"]) for r in sel],
                               [light(inverse_normalize(hy[r["id"]]) if spoken else hy[r["id"]]) for r in sel])
        errs = sum("error" in x for x in load(f) if x["id"] in ids)
        print(f"| {name} | {res['all'][1]:.2f} | {res['no-digit'][1]:.2f} | {fair:.2f} |" + (f" ({errs} load errors)" if errs else ""))
rtf = sorted(glob.glob(str(STT / "eval/bench_rtf_*.json")))
if rtf:
    print("\n## CPU speed (1 thread, fp32): CPU seconds per second of audio (FLEURS kk utterances)\n")
    print("| system | CPU s / audio s | s per utterance |\n|---|---|---|")
    for f in rtf:
        d = json.load(open(f))
        print(f"| {d['tag']} ({d['threads']} thr) | {d['cpu_s_per_audio_s']} | {d['sec_per_utt']} |")
