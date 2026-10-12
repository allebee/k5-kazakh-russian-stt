"""WiSE-FT: write checkpoints with weights alpha*finetuned + (1-alpha)*base, loadable by gigaam.load_model().

usage: wise_ft.py FINETUNED.ckpt BASE_NAME OUT_DIR [alpha ...]   (default alphas 0.25 0.5 0.75)
"""
import os, sys
from pathlib import Path

import torch

import gigaam

gigaam._CACHE_DIR = os.path.join(os.path.abspath(os.environ.get("K5_STT_ROOT", ".")), ".gigaam-cache")
ft_path, base_name, out_dir = sys.argv[1:4]
alphas = [float(a) for a in sys.argv[4:]] or [0.25, 0.5, 0.75]
PREFIX = ("preprocessor.", "encoder.", "head.")

ft = torch.load(ft_path, map_location="cpu", weights_only=False)
base = gigaam.load_model(base_name, fp16_encoder=False, device="cpu").state_dict()
keys = [k for k in ft["state_dict"] if k.startswith(PREFIX)]
missing = [k for k in keys if k not in base]
assert not missing, f"keys missing in base: {missing[:5]}"
Path(out_dir).mkdir(parents=True, exist_ok=True)
for a in alphas:
    sd = dict(ft["state_dict"])
    for k in keys:
        t = sd[k]
        if t.is_floating_point():
            sd[k] = (a * t.float() + (1 - a) * base[k].float()).to(t.dtype)
    out = Path(out_dir) / f"wise_a{a:.2f}.ckpt"
    torch.save({"state_dict": sd, "hyper_parameters": ft["hyper_parameters"]}, out)
    print(f"alpha {a:.2f} -> {out}")
