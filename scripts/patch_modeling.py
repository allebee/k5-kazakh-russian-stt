"""Make pyannote optional in modeling_gigaam.py (ai-sage/GigaAM-Multilingual, MIT): transformers' remote-code import
check refuses to load a model whose modeling file mentions a missing package, and pyannote is only needed for
transcribe_longform (VAD segmentation). Static `from pyannote.X import A, B` lines become dynamic imports with the
same names, so behaviour is unchanged when pyannote is installed and the model loads without it otherwise.
usage: patch_modeling.py FILE [FILE ...]   (keeps FILE.orig)"""
import re, shutil, sys

HELPER = '''

def _optional_import(module, *names):
    """Import names from an optional package at call time (pyannote: needed only for transcribe_longform)."""
    import importlib
    m = importlib.import_module(module)
    vals = tuple(getattr(m, n) for n in names)
    return vals if len(vals) > 1 else vals[0]
'''
pat = re.compile(r"^(\s*)from (pyannote[\w.]*) import ([\w, ]+)$", re.M)
for f in sys.argv[1:]:
    src = open(f, encoding="utf-8").read()
    if "_optional_import" in src:
        print(f, "already patched"); continue
    shutil.copy(f, f + ".orig")
    n = len(pat.findall(src))
    out = pat.sub(lambda m: f"{m.group(1)}{m.group(3).strip()} = _optional_import({m.group(2)!r}, "
                            + ", ".join(repr(x.strip()) for x in m.group(3).split(",")) + ")", src)
    # helper right after the last top-level import block
    i = max(out.rfind("\nimport "), out.rfind("\nfrom "))
    j = out.index("\n", i + 1)
    out = out[:j + 1] + HELPER + out[j + 1:]
    open(f, "w", encoding="utf-8").write(out)
    print(f, f"patched {n} pyannote import lines")
