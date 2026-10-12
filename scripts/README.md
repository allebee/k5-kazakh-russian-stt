# Release tools

| File | What it does |
|---|---|
| `convert_to_hf.py` | `convert`: training `.ckpt` → `model.safetensors` (keys = `"model."` + GigaAM state-dict key, fp32, optimizer state dropped) + `config.json` + `modeling_gigaam.py` from GigaAM-Multilingual `large_ctc`, for the offline model (repo root) and the streaming model (`streaming/`); also writes reference transcripts. `verify`: loads both with `transformers.AutoModel` and checks the text is identical. Paths: `$K5_STT_ROOT`, `$HF_STAGE_DIR` |
| `patch_modeling.py` | makes the `pyannote` imports in `modeling_gigaam.py` optional (only `transcribe_longform` needs it), so the model loads with `trust_remote_code` without pyannote installed |
| `release_test.py` | end-to-end test of a downloaded/staged model repo in a clean environment: both models through `AutoModel`, the two-pass pipeline from the repo's `deploy/`, ITN and hotwords on dev utterances |

`modeling_gigaam.py` itself is MIT (© 2024 GigaChat Team) and ships with the model on Hugging Face, not here.
