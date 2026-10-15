"""Turn the released Hugging Face weights back into a GigaAM training checkpoint, to continue fine-tuning.

The HF repo alibiserikbay/k5-kazakh-russian-stt stores each model as model.safetensors with keys "model." + GigaAM
state_dict keys (fp32, no optimizer state; see convert_to_hf.py). GigaAM's trainer (train_utils/train.py, wrapped by
training/train_wrapper.py) loads a .ckpt with {"cfg", "state_dict"}. This script takes the base GigaAM-Multilingual
large_ctc checkpoint (same architecture), swaps in our weights after checking that every key and shape matches, and saves
a .ckpt that train_wrapper.py accepts via --model_name (as training/train_streaming.sh does with STAGE_INIT).

usage (environment with the gigaam package + huggingface_hub + safetensors):
  python scripts/hf_to_gigaam_ckpt.py [offline|streaming] [OUT.ckpt]
  # offline (default) -> $K5_STT_ROOT/runs/full_multilingual_large_ctc/final_a1.00.ckpt  (what train_streaming.sh expects)
  # streaming         -> $K5_STT_ROOT/runs/stream_multilingual_large_ctc/final_stream.ckpt
Then continue training, e.g. STAGE_INIT=<that ckpt> or link it into $GIGAAM_DIR/train_utils/ and pass --model_name.
"""
import os, sys
from pathlib import Path

import torch
import gigaam
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO = os.environ.get("K5_STT_HF_REPO", "alibiserikbay/k5-kazakh-russian-stt")
STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()
which = sys.argv[1] if len(sys.argv) > 1 else "offline"
assert which in ("offline", "streaming"), "first argument: offline | streaming"
default_out = {"offline": STT / "runs/full_multilingual_large_ctc/final_a1.00.ckpt",
               "streaming": STT / "runs/stream_multilingual_large_ctc/final_stream.ckpt"}[which]
out = Path(sys.argv[2]) if len(sys.argv) > 2 else default_out

# 1. our released weights
fn = "model.safetensors" if which == "offline" else "streaming/model.safetensors"
ours = load_file(hf_hub_download(REPO, fn))
sd = {k[len("model."):]: v for k, v in ours.items() if k.startswith("model.")}
assert sd, f"no 'model.' keys in {fn}"

# 2. the base checkpoint (downloaded into the gigaam cache on first use), for its cfg and key layout
gigaam._CACHE_DIR = str(STT / ".gigaam-cache")
gigaam.load_model("multilingual_large_ctc", device="cpu")          # makes sure the base .ckpt is in the cache
base_path = next(Path(gigaam._CACHE_DIR).glob("multilingual_large_ctc*.ckpt"))
base = torch.load(base_path, map_location="cpu", weights_only=False)
bsd = base["state_dict"]

# 3. strict check: same keys, same shapes
missing, extra = sorted(set(bsd) - set(sd)), sorted(set(sd) - set(bsd))
bad_shape = [k for k in set(bsd) & set(sd) if tuple(bsd[k].shape) != tuple(sd[k].shape)]
if missing or extra or bad_shape:
    sys.exit(f"layout mismatch: missing {missing[:5]} extra {extra[:5]} shape {bad_shape[:5]}")

base["state_dict"] = {k: sd[k].to(bsd[k].dtype) for k in bsd}
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(base, out)
print(f"saved {out} ({len(sd)} tensors from {REPO}/{fn}, cfg from {base_path.name})")

# 4. sanity check: the rebuilt checkpoint loads and transcribes
if os.environ.get("K5_CHECK_WAV"):
    m = gigaam.load_model(str(out), device="cpu")
    print("transcript:", m.transcribe(os.environ["K5_CHECK_WAV"]))
