"""End-of-turn classifiers at the `pause` event, on the 100 simulated dev calls (3 dev utterances joined with 0.9-1.3 s
of silence, as in stream_session_test.py).
Every `pause` (192 ms of VAD silence) is labelled complete = no VAD speech before the end of the same utterance after
it (the speaker finished the sentence), else incomplete (a pause inside the sentence). Classifiers:
  heuristic   looks_final() on the pause text (kk particles / finite-verb endings / closers), as in StreamSession
  smart_turn  Pipecat Smart Turn v3.x ONNX on the last <= 8 s of the turn audio up to the pause (192 ms silence at the
              end), left-padded with zeros to 8 s as in their inference code, P(complete) > 0.5
  feanet      feanet/eou-detector-russian (Qwen2.5-0.5B): P(<|im_end|>) after the pause text as a user turn, > 0.077
Stage 1 (service env):  turn_eval.py collect OUT.jsonl   env: CALLS=100 PROCS=16 THREADS=2 EOU_MS=800
Stage 2 (transformers): turn_eval.py score IN.jsonl OUT.json   env: SMART_TURN=path.onnx FEANET=dir"""
import json, os, random, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "service")]
STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace with eval/manifests/dev_*.jsonl
SETS = ("dev_fleurs_kk", "dev_fleurs_ru", "dev_ksc2_spont")
LANG = {"dev_fleurs_kk": "kk_read", "dev_fleurs_ru": "ru_read", "dev_ksc2_spont": "kk_spont"}


def load_calls(n):
    """Same audio, order and gaps as stream_session_test.load_calls, plus each utterance's span and source set."""
    import soundfile as sf
    rows = [dict(json.loads(l), _set=s) for s in SETS for l in open(STT / f"eval/manifests/{s}.jsonl", encoding="utf-8")]
    random.Random(1).shuffle(rows)
    calls = []
    for i in range(n):
        wavs, utts, t = [], [], 0.0
        for r in rows[i * 3:(i + 1) * 3]:
            w, sr = sf.read(r["path"], dtype="float32")
            w = w if w.ndim == 1 else w.mean(1)
            gap = int(16000 * random.Random(i).uniform(0.9, 1.3))
            wavs += [w, np.zeros(gap, np.float32)]
            utts.append({"start": t, "end": t + len(w) / 16000, "set": r["_set"]}); t += (len(w) + gap) / 16000
        calls.append({"wav": np.concatenate(wavs).astype(np.float32), "utts": utts})
    return calls


def collect_call(job):
    from stream_session_test import M
    from twopass_stream import StreamSession, looks_final
    k, call = job
    s = StreamSession(M["tp"], eou_ms=int(os.environ.get("EOU_MS", "800")), pause_ms=200, lang="beam")
    wav, ev = call["wav"], []
    for i in range(0, len(wav), 320):
        e = s.feed(wav[i:i + 320]); ev += e
        if e and e[-1]["type"] == "eou":
            ev += s.poll()
    ev += s.flush()
    from twopass_stream import W
    onsets = sorted(e["t"] if e["type"] == "speech_start" else e["t"] - W for e in ev if e["type"] in ("speech_start", "resume"))
    out, turn0 = [], None
    for e in ev:
        if e["type"] == "speech_start":
            turn0 = e["t"]
        if e["type"] != "pause":
            continue
        te = e["t_speech_end"]
        u = next((u for u in call["utts"] if u["start"] - 0.1 <= te <= u["end"] + 0.1), None)
        if u is None:
            continue
        complete = not any(te + 0.05 < o < u["end"] - 0.05 for o in onsets)
        nxt = next((o for o in onsets if o > te + 0.05), None)
        out.append({"call": k, "t": e["t"], "t_speech_end": te, "turn0": turn0, "text": e["text"], "set": u["set"],
                    "silence": None if nxt is None else round(nxt - te, 3),
                    "lang": LANG[u["set"]], "complete": complete, "heuristic": looks_final(e["text"])})
    return out


def collect(path, n):
    from concurrent.futures import ProcessPoolExecutor
    from stream_session_test import init
    calls = load_calls(n)
    with ProcessPoolExecutor(int(os.environ.get("PROCS", "16")), initializer=init) as ex:
        res = list(ex.map(collect_call, list(enumerate(calls))))
    with open(path, "w", encoding="utf-8") as f:
        for r in res:
            for p in r:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{sum(map(len, res))} pauses", flush=True)


def prf(y, yhat):
    y, yhat = np.asarray(y, bool), np.asarray(yhat, bool)
    tp = int((y & yhat).sum()); p = tp / max(1, yhat.sum()); r = tp / max(1, y.sum())
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3),
            "fpr": round(int((~y & yhat).sum()) / max(1, (~y).sum()), 3), "n_pos_pred": int(yhat.sum())}


def auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return None
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return round(float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 3)


def score(inp, outp):
    import time
    import onnxruntime as ort
    import soundfile  # noqa: F401  (load_calls)
    from transformers import WhisperFeatureExtractor
    P = [json.loads(l) for l in open(inp, encoding="utf-8")]
    calls = load_calls(1 + max(p["call"] for p in P))
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
    st = ort.InferenceSession(os.environ["SMART_TURN"], o)
    fe = WhisperFeatureExtractor(chunk_length=8)
    ms = []
    for p in P:
        wav = calls[p["call"]]["wav"]
        a0 = max(0.0, (p["turn0"] if p["turn0"] is not None else 0.0) - 0.2, p["t"] - 8.0)
        x = wav[int(a0 * 16000):int(p["t"] * 16000)][-8 * 16000:]
        x = np.pad(x, (8 * 16000 - len(x), 0))          # zero-pad at the BEGINNING, as smart-turn audio_utils.py does
        f = fe(x, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=8 * 16000,
               truncation=True, do_normalize=True).input_features.squeeze(0).astype(np.float32)[None]
        t0 = time.perf_counter()
        p["smart_turn_p"] = float(np.ravel(st.run(None, {"input_features": f})[0])[0])
        ms.append(1000 * (time.perf_counter() - t0))
    lat = {"smart_turn_ms_median": round(float(np.median(ms)), 1)}
    if os.environ.get("FEANET"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(int(os.environ.get("THREADS", "4")))
        tok = AutoTokenizer.from_pretrained(os.environ["FEANET"])
        lm = AutoModelForCausalLM.from_pretrained(os.environ["FEANET"], torch_dtype=torch.float32).eval()
        eid = tok.convert_tokens_to_ids("<|im_end|>"); ms = []
        for p in P:
            prompt = tok.apply_chat_template([{"role": "user", "content": p["text"]}], add_generation_prompt=False,
                                             tokenize=False)
            prompt = prompt[:prompt.rfind("<|im_end|>")]
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
            t0 = time.perf_counter()
            with torch.no_grad():
                p["feanet_p"] = float(torch.softmax(lm(**enc).logits[0, -1], -1)[eid])
            ms.append(1000 * (time.perf_counter() - t0))
        lat["feanet_ms_median"] = round(float(np.median(ms)), 1)
    with open(inp.replace(".jsonl", ".scored.jsonl"), "w", encoding="utf-8") as f:
        for p in P:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    groups = {"all": P, "ru_read": [p for p in P if p["lang"] == "ru_read"],
              "kk_all": [p for p in P if p["lang"] != "ru_read"], "kk_read": [p for p in P if p["lang"] == "kk_read"],
              "kk_spont": [p for p in P if p["lang"] == "kk_spont"]}
    out = {"latency": lat}
    for g, ps in groups.items():
        y = [p["complete"] for p in ps]
        d = {"pauses": len(ps), "complete": int(sum(y)), "base_rate": round(float(np.mean(y)), 3),
             "heuristic": prf(y, [p["heuristic"] for p in ps]),
             "smart_turn": {**prf(y, [p["smart_turn_p"] > 0.5 for p in ps]), "auc": auc(y, [p["smart_turn_p"] for p in ps])}}
        if "feanet_p" in ps[0]:
            d["feanet"] = {**prf(y, [p["feanet_p"] > 0.077 for p in ps]), "auc": auc(y, [p["feanet_p"] for p in ps])}
            d["smart_turn_and_feanet"] = prf(y, [p["smart_turn_p"] > 0.5 and p["feanet_p"] > 0.077 for p in ps])
        # adaptive EOU policy: wait 400 ms if the classifier says complete, else 1200 ms. A pause inside
        # a sentence that lasts >= the wait cuts the turn; at a true end the wait is the added latency
        def policy(dec, yes=400, no=1200):
            cut = sum(1 for p, c in zip(ps, dec) if not p["complete"] and (p["silence"] or 9) * 1000 >= (yes if c else no))
            wait = [yes if c else no for p, c in zip(ps, dec) if p["complete"]]
            return {"cuts_per_100_ends": round(100 * cut / max(1, sum(y)), 1), "end_wait_ms_mean": round(float(np.mean(wait)), 0)}
        d["policy"] = {"fixed400": policy([True] * len(ps)), "fixed700": policy([True] * len(ps), yes=700),
                       "fixed1200": policy([False] * len(ps)), "heuristic": policy([p["heuristic"] for p in ps]),
                       "smart_turn": policy([p["smart_turn_p"] > 0.5 for p in ps]),
                       "smart_turn_and_heuristic": policy([p["smart_turn_p"] > 0.5 and p["heuristic"] for p in ps])}
        if "feanet_p" in ps[0]:
            d["policy"]["feanet"] = policy([p["feanet_p"] > 0.077 for p in ps])
            d["policy"]["smart_turn_and_feanet"] = policy([p["smart_turn_p"] > 0.5 and p["feanet_p"] > 0.077 for p in ps])
        # best-F1 threshold for Smart Turn (oracle, optimistic: shows the ceiling of a recalibration)
        best = max(((prf(y, [p["smart_turn_p"] > th for p in ps])["f1"], th) for th in np.arange(0.05, 0.96, 0.05)))
        d["smart_turn_best_f1_oracle"] = {"thr": round(float(best[1]), 2), **prf(y, [p["smart_turn_p"] > best[1] for p in ps])}
        out[g] = d
        print(g, json.dumps(d), flush=True)
    json.dump(out, open(outp, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2], int(os.environ.get("CALLS", "100")))
    else:
        score(sys.argv[2], sys.argv[3])
