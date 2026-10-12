"""Stage the release in the ai-sage/GigaAM-Multilingual format (NO upload): for each model, model.safetensors (keys =
"model." + our state_dict keys, fp32, optimizer state dropped) + config.json + modeling_gigaam.py from the large_ctc
revision (MIT, unchanged), so `AutoModel.from_pretrained(dir, trust_remote_code=True).transcribe(wav)` works.
Step 1 (environment with the gigaam package): convert + reference transcripts with gigaam.load_model(ckpt).
Step 2 (environment with transformers): load each staged dir with AutoModel and compare transcripts.
usage: convert_to_hf.py convert|verify"""
import json, os, shutil, sys
from pathlib import Path

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace: data/, manifests/, eval/, runs/, lm/, models/
OUT = Path(os.environ.get("HF_STAGE_DIR", STT / "hf_release/k5-kazakh-russian-stt"))   # staging dir = the repo layout
SRC = STT / "models/gigaam-ml-large-ctc"
MODELS = {".": (STT / "runs/full_multilingual_large_ctc/final_a1.00.ckpt", "k5_kazakh_russian_large_ctc"),
          "streaming": (STT / "runs/stream_multilingual_large_ctc/final_stream.ckpt", "k5_kazakh_russian_large_ctc_streaming")}
WAVS = [json.loads(l)["path"] for s in ("dev_fleurs_kk", "dev_fleurs_ru", "dev_ksc2_spont")
        for l in list(open(STT / f"eval/manifests/{s}.jsonl", encoding="utf-8"))[:2]]

if sys.argv[1] == "convert":
    import torch, gigaam
    from safetensors.torch import save_file
    gigaam._CACHE_DIR = str(STT / ".gigaam-cache")
    ref = {}
    for name, (ckpt, mname) in MODELS.items():
        d = OUT / name; d.mkdir(parents=True, exist_ok=True)
        if not (d / "model.safetensors").exists():
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
            save_file({"model." + k: v.contiguous() for k, v in sd.items()}, str(d / "model.safetensors"), metadata={"format": "pt"})
        cfg = json.load(open(SRC / "config.json"))
        cfg["cfg"]["model"]["cfg"]["model_name"] = mname
        json.dump(cfg, open(d / "config.json", "w"), indent=2)
        shutil.copy(SRC / "modeling_gigaam.py", d / "modeling_gigaam.py")
        m = gigaam.load_model(str(ckpt), fp16_encoder=False, device="cpu").eval()
        ref[name] = [getattr(r, "text", r) for r in (m.transcribe(w) for w in WAVS)]
        print(name, "->", d, f"{(d / 'model.safetensors').stat().st_size / 1e9:.2f} GB", flush=True)
    json.dump(ref, open(OUT.parent / "reference_transcripts.json", "w"), ensure_ascii=False, indent=1)
else:
    import torch
    from transformers import AutoModel
    ref = json.load(open(OUT.parent / "reference_transcripts.json"))
    for name in MODELS:
        m = (AutoModel.from_pretrained(str(OUT), trust_remote_code=True) if name == "." else
             AutoModel.from_pretrained(str(OUT), subfolder=name, trust_remote_code=True)).eval()
        hyp = [m.transcribe(w) for w in WAVS]
        hyp = [h if isinstance(h, str) else getattr(h, "text", str(h)) for h in hyp]
        rr = [r if isinstance(r, str) else getattr(r, "text", str(r)) for r in ref[name]]
        same = sum(a == b for a, b in zip(hyp, rr))
        print(f"{'offline (root)' if name == '.' else name}: AutoModel vs gigaam.load_model identical {same}/{len(hyp)}")
        for a, b in zip(hyp, rr):
            if a != b:
                print("  HF:", a[:100], "\n  gigaam:", b[:100])
