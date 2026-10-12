"""Run every reference case through kk_tn and report pass rates per category.

    python tests/run_cases.py            # summary + failures, writes tests/REPORT.md

Comparison is lenient: lowercase, punctuation and hyphens dropped, whitespace collapsed.
Cases listed in tests/data/disputes.tsv are reported separately: the rule follows the
user's convention and the case contradicts another approved case (or vice versa).
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from kk_tn import normalize  # noqa: E402

DATA = HERE / "data"


def canon(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[-–—]", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_cases():
    cases = []
    with open(DATA / "tn_itn_cases.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["direction"] != "TN":
                continue
            alts = re.findall(r"alt: ([^;)]+)", r["note"])
            cases.append(dict(id=f"r{int(r['id']):03d}", src="research", category=r["category"],
                              written=r["written"], expected=r["expected_spoken"], alts=alts))
    with open(DATA / "tn_cases_tts_v0.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            alts = re.findall(r"alt: ([^;)]+)", r.get("note") or "")
            cases.append(dict(id=r["id"], src="tts", category=r["category"], written=r["written"],
                              expected=r["expected_spoken"], alts=alts))
    with open(DATA / "codeswitch_kkru_v0.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            cases.append(dict(id=r["id"], src="codeswitch", category="passthrough_cs",
                              written=r["text"], expected=r["text"], alts=[]))
    return cases


def load_disputes() -> dict[str, str]:
    p = DATA / "disputes.tsv"
    if not p.exists():
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t"):
            if r and not r[0].startswith("#"):
                out[r[0]] = r[1] if len(r) > 1 else ""
    return out


def run_itn(lines: list[str]) -> int:
    from kk_tn.itn import inverse_normalize
    rows = []
    with open(DATA / "tn_itn_cases.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["direction"] == "ITN":
                rows.append(r)
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()  # noqa: E731
    fails = []
    for r in rows:
        got = inverse_normalize(r["expected_spoken"])
        if norm(got) != norm(r["written"]):
            fails.append((r, got))
    lines += ["", "## ITN kk / ru / mixed (spoken → written)", "",
              f"{len(rows)} cases · pass {len(rows) - len(fails)} · fail {len(fails)}", ""]
    if fails:
        lines += ["| id | lang | spoken | got | expected |", "|---|---|---|---|---|"]
        lines += [f"| {r['id']} | {r['lang']} | {r['expected_spoken']} | {got} | {r['written']} |" for r, got in fails]
    return len(fails)


def main() -> int:
    cases, disputes = load_cases(), load_disputes()
    stats = defaultdict(lambda: [0, 0, 0, 0])  # pass, alt, dispute, fail
    fails, dispute_rows = [], []
    for c in cases:
        got = normalize(c["written"])
        ok = canon(got) == canon(c["expected"])
        alt = not ok and any(canon(got) == canon(a) for a in c["alts"])
        s = stats[c["category"]]
        if ok:
            s[0] += 1
        elif c["id"] in disputes:
            s[2] += 1
            dispute_rows.append((c, got, disputes[c["id"]]))
        elif alt:
            s[1] += 1
        else:
            s[3] += 1
            fails.append((c, got))
    tot = [sum(v[i] for v in stats.values()) for i in range(4)]
    n = sum(tot)
    lines = ["# kk_tn v0 test report", "",
             f"{n} cases · pass {tot[0]} · pass-on-alt {tot[1]} · dispute {tot[2]} · fail {tot[3]} · "
             f"strict pass rate {tot[0] / n:.1%} · excluding disputes {(tot[0] + tot[1]) / max(1, n - tot[2]):.1%}",
             "", "| category | pass | alt | dispute | fail |", "|---|---|---|---|---|"]
    for cat in sorted(stats):
        p, a, d, f = stats[cat]
        lines.append(f"| {cat} | {p} | {a} | {d} | {f} |")
    lines += ["", "## Disputes (rule vs case)", "", "| id | written | got | expected | why |", "|---|---|---|---|---|"]
    for c, got, why in dispute_rows:
        lines.append(f"| {c['id']} | {c['written']} | {got} | {c['expected']} | {why} |")
    lines += ["", "## Failures", "", "| id | category | written | got | expected |", "|---|---|---|---|---|"]
    for c, got in fails:
        lines.append(f"| {c['id']} | {c['category']} | {c['written']} | {got} | {c['expected']} |")
    itn_fails = run_itn(lines)
    report = "\n".join(lines) + "\n"
    (HERE / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    return 0 if not fails and not itn_fails else 1


if __name__ == "__main__":
    sys.exit(main())
