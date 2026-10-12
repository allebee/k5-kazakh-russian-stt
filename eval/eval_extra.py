"""Extra held-out eval sets (JSONL for the harness):

  ksc2_<source>   KSC2 Test per source (crowdsourced, tts capped at 500 random; spontaneous sources complete).
  cs_ksc2         natural kk<->ru code-switched utterances mined from KSC2 Dev+Test (not in training).
                  An utterance counts if it has >=2 Russian-marked words (at least one of length >=4) and >=2
                  Kazakh-marked words. Russian-marked: seen >=3 times in ru training text, no Kazakh-specific letter,
                  and >=5x more frequent (per token) in ru than in kk training text. Kazakh-marked: contains әғқңөұүһі.
                  These utterances are also removed from manifests/val.tsv, so model selection never sees them.
  golos_crowd / golos_farfield   Golos test (crowd capped at 1000 random).
"""
import collections, glob, json, os, random, re
from pathlib import Path

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
KSC2 = Path(os.environ.get("KSC2_DIR", STT / "data/ksc2/extracted/ISSAI_KSC2"))
MAN, MT = STT / "eval/manifests", STT / "manifests"
KK_LETTERS = set("әғқңөұүһі")
random.seed(5)
_PUNCT = re.compile(r"[^\w\s']|_")


def words(text):
    t = text.lower().replace("ё", "е")
    t = re.sub(r"[‐-―\-]", " ", t)
    return _PUNCT.sub(" ", t).split()


def write(name, rows):
    with open(MAN / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)}", flush=True)


def ksc2_rows(split):
    out = []
    for txt in sorted(glob.glob(str(KSC2 / split / "*" / "*.txt"))):
        p = Path(txt)
        stem = p.name.split(".")[0]  # some KSC2 transcripts are named <id>.txt.txt
        audio = p.parent / f"{stem}.flac"
        if not audio.exists():
            continue
        out.append({"id": f"ksc2_{split}_{p.parent.name}_{stem}", "path": str(audio),
                    "ref": p.read_text(encoding="utf-8").strip(), "lang": "kk", "source": p.parent.name})
    return out


def counts(names):
    c = collections.Counter()
    for n in names:
        for line in open(MT / f"{n}.tsv", encoding="utf-8").read().splitlines()[1:]:
            c.update(line.split("\t")[2].split())
    return c


if __name__ == "__main__":
    test = ksc2_rows("Test")
    by = collections.defaultdict(list)
    for r in test:
        by[r["source"]].append(r)
    for src, rows in sorted(by.items()):
        if src in ("crowdsourced", "tts") and len(rows) > 500:
            rows = random.sample(rows, 500)
        write(f"ksc2_{src}", rows)

    ru, kk = counts(["ru_golos", "ru_sova", "ru_ruls"]), counts(["kk_ksc2", "kk_ksd"])
    nru, nkk = sum(ru.values()), sum(kk.values())

    def ru_marked(w):
        return (ru[w] >= 3 and not (set(w) & KK_LETTERS)
                and ru[w] / nru >= 5 * (kk[w] + 0.5) / nkk)

    cs = []
    for r in ksc2_rows("Dev") + test:
        ws = words(r["ref"])
        rw = [w for w in ws if ru_marked(w)]
        kw = [w for w in ws if set(w) & KK_LETTERS]
        if len(rw) >= 2 and any(len(w) >= 4 for w in rw) and len(kw) >= 2:
            cs.append(dict(r, lang="kkru", ru_words=rw))
    print(f"code-switched candidates in KSC2 Dev+Test: {len(cs)} "
          f"(by source: {dict(collections.Counter(r['source'] for r in cs))})", flush=True)
    cs = random.sample(cs, min(600, len(cs)))
    write("cs_ksc2", cs)
    held = {r["path"] for r in cs}
    lines = open(MT / "val.tsv", encoding="utf-8").read().splitlines()
    kept = [lines[0]] + [l for l in lines[1:] if l.split("\t")[0] not in held]
    open(MT / "val.tsv", "w", encoding="utf-8").write("\n".join(kept) + "\n")
    print(f"val.tsv: removed {len(lines) - len(kept)} code-switched rows", flush=True)

    for kind, cap in (("crowd", 1000), ("farfield", 0)):
        mans = glob.glob(str(STT / f"data/golos/test/{kind}/*.jsonl")) + glob.glob(str(STT / f"data/golos/golos_opus/test/{kind}/*.jsonl"))
        if not mans:
            print(f"golos {kind}: no manifest found", flush=True)
            continue
        base = Path(mans[0]).parent
        rows = []
        for i, r in enumerate(map(json.loads, open(mans[0], encoding="utf-8"))):
            rows.append({"id": f"golos_{kind}_{i}", "path": str(base / r["audio_filepath"]), "ref": r["text"], "lang": "ru"})
        if cap and len(rows) > cap:
            rows = random.sample(rows, cap)
        write(f"golos_{kind}", rows)
    print("EVAL_EXTRA_DONE", flush=True)
