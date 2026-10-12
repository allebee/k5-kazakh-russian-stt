"""1-thread CPU speed of our offline FT model (GigaAM-ML 600M CTC, greedy) on the same 6 FLEURS kk bench utterances."""
import json, os, time, sys
from pathlib import Path
import torch, gigaam
STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
torch.set_num_threads(1)
gigaam._CACHE_DIR = str(STT / ".gigaam-cache")
m = gigaam.load_model(os.environ.get("F_CKPT", str(STT / "runs/full_multilingual_large_ctc/final_a1.00.ckpt")), fp16_encoder=False, device="cpu").eval()
rows = [json.loads(l) for l in open(STT / "eval/manifests/bench_fleurs_kk.jsonl", encoding="utf-8")][:6]
def run(r):
    t0 = time.perf_counter()
    with torch.inference_mode():
        wav, ln = m.prepare_wav(r["path"]); enc, _ = m.forward(wav, ln); m.head(enc)
    return time.perf_counter() - t0, ln.item() / 16000
run(rows[0])
res = [run(r) for r in rows]
sec, aud = sum(a for a, _ in res), sum(b for _, b in res)
d = {"tag": "ours-FT-600M-greedy", "threads": 1, "cpu_s_per_audio_s": round(sec / aud, 3), "sec_per_utt": round(sec / len(res), 2), "utts": len(res)}
json.dump(d, open(STT / "eval/bench_rtf_ours_t1.json", "w")); print(d)
