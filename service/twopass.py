"""Two-pass streaming STT for Kazakh, Russian and mixed kk/ru speech (CPU or GPU).

  audio --> VAD (Silero vad.onnx + AGC) --> segment [speech start - PAD, speech end + PAD]
              |                                         |
              +--> partials: chunk model, cache engine  +--> final: offline fine-tune, full context,
                   (R=0, chunk 320/640 ms), every chunk      KenLM beam (kkru by default) + kk_tn ITN

- Partials: committed chunks never change; the still-growing last word is hidden unless tentative=True.
- Endpoint: EOU s of silence (default 0.5). A segment is force-cut at MAX_SEG s (the models saw <= ~30 s).
- Live segmentation: the next segment's pre-roll is clamped to start where the previous padded segment ended, so no
  audio is decoded twice and no final is ever revised.
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SR = 16000


def release_paths(root):
    """Paths inside the released repo (k5-kazakh-russian-stt): offline model at the root, chunk model in streaming/."""
    r = Path(root)
    return {"partial": r / "streaming", "final": r, "vad": r / "vad/vad.onnx", "lm_dir": r / "lm",
            "kk_tn": r / "deploy/kk_tn", "lm_weights": r / "lm/lm_weights.json", "gigaam_cache": None}


def _load(path, device):
    """HF format (config.json + modeling_gigaam.py, trust_remote_code) or a gigaam .ckpt."""
    import torch
    path = Path(path)
    if path.is_dir() and (path / "config.json").exists():
        from transformers import AutoModel
        return AutoModel.from_pretrained(str(path), trust_remote_code=True).model.to(device).eval()
    import gigaam
    return gigaam.load_model(str(path), fp16_encoder=device.type == "cuda", device=device).eval()


def default_paths(stt_root=None):
    """Training-side layout (GigaAM .ckpt files written by training/) under the workspace $K5_STT_ROOT (default: the
    current directory). For the released Hugging Face repo use release_paths(<downloaded dir>) instead."""
    s = Path(stt_root or os.environ.get("K5_STT_ROOT", ".")).resolve()
    return {"partial": s / "runs/stream_multilingual_large_ctc/final_stream.ckpt",           # C v1 (R=0)
            "final": s / "runs/full_multilingual_large_ctc/final_a1.00.ckpt",               # offline fine-tune
            "vad": s / "models/old/vad/vad.onnx", "lm_dir": s / "lm", "kk_tn": HERE.parent / "kk_tn",
            "lm_weights": s / "eval/lm_ft-large_ctc-raw.json", "gigaam_cache": s / ".gigaam-cache"}


class StreamVAD:
    """Silero VAD (vad.onnx, as shipped in the model repo's vad/) on 96 ms windows, with AGC (running RMS -> -20 dBFS, max x30): without
    AGC it misses quiet recordings (~-36 dBFS peak). Events: ("start", t) after >= MIN_SPEECH s of speech prob > THR;
    ("end", t_end) after EOU s below THR. Times are in seconds of stream audio."""
    WIN = 1536

    def __init__(self, path, thr=0.5, eou=0.5, min_speech=0.25):
        import onnxruntime as ort
        o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), o)
        self.thr, self.eou, self.min_speech = thr, eou, min_speech
        self.h = self.c = np.zeros((2, 1, 64), np.float32)
        self.buf, self.k, self.rms = np.zeros(0, np.float32), 0, None
        self.run = self.sil = 0
        self.in_speech = False

    def feed(self, x):
        w, ev = self.WIN / SR, []
        self.buf = np.concatenate([self.buf, x])
        while len(self.buf) >= self.WIN:
            win, self.buf = self.buf[:self.WIN], self.buf[self.WIN:]
            r = float(np.sqrt(np.mean(win ** 2)) + 1e-6)
            self.rms = r if self.rms is None else (0.9 * self.rms + 0.1 * r if r < self.rms else 0.5 * self.rms + 0.5 * r)
            win = np.clip(win * min(30.0, 0.1 / self.rms), -1, 1).astype(np.float32)
            p, self.h, self.c = self.sess.run(None, {"input": win[None], "sr": np.array(SR, np.int64), "h": self.h, "c": self.c})
            t = self.k * w; self.k += 1
            if float(np.ravel(p)[0]) > self.thr:
                self.run += 1; self.sil = 0
                if not self.in_speech and self.run * w >= self.min_speech:
                    self.in_speech = True; ev.append(("start", t - (self.run - 1) * w))
            else:
                self.run = 0
                if self.in_speech:
                    self.sil += 1
                    if self.sil * w >= self.eou:
                        self.in_speech = False; ev.append(("end", t - (self.sil - 1) * w)); self.sil = 0
        return ev


class TwoPass:
    def __init__(self, device="cpu", chunk_ms=320, pad=0.5, eou=0.5, max_seg=20.0, lm="kkru", itn=True,
                 tentative=False, threads=4, paths=None, beam_width=64, bf16_final=True, bf16_partials=False,
                 hotwords=None, hotword_weight=10.0):
        """CPU precision (AMX bf16, measured 2026-10-08 on a Xeon 8570, 4 threads): the final pass is ~2x faster in bf16
        (10 s segment 504 -> 246 ms) with WER parity (dev +-0.1; live finals 9.03 vs 9.19), so it is on by default.
        The partial engine stays fp32 by default: bf16 is slower at 320 ms chunks (147 vs 107 ms) and only ~20%
        faster at 640 ms (152 vs 193 ms)."""
        import streaming_patch
        self.p = paths or default_paths()
        torch.set_num_threads(threads)
        if self.p.get("gigaam_cache"):
            import gigaam
            gigaam._CACHE_DIR = str(self.p["gigaam_cache"])
        self.dev = torch.device(device)
        self.cm = _load(self.p["partial"], self.dev)
        self.fm = _load(self.p["final"], self.dev)
        for m in (self.cm, self.fm):  # patch the module that defines each encoder (gigaam package or HF remote code)
            mod = sys.modules[type(m.encoder).__module__]
            streaming_patch.enable(None if mod.__name__.startswith("gigaam") else mod)
        self.fm.encoder._force_full = True
        self.vocab = [str(v) for v in self.fm.decoding.tokenizer.vocab]
        self.C, self.pad, self.max_seg, self.tentative = chunk_ms // 40, pad, max_seg, tentative
        cpu = self.dev.type == "cpu"
        self.bf16_final, self.bf16_partials = cpu and bf16_final, cpu and bf16_partials
        self.eou = eou
        self.dec = None
        if lm and lm != "none":
            from pyctcdecode import build_ctcdecoder
            a, b = (json.load(open(self.p["lm_weights"])).get(lm) or [0.2, 0.0])
            uni = Path(self.p["lm_dir"]) / f"{lm}.unigrams.txt"
            self.dec = build_ctcdecoder(self.vocab + [""], kenlm_model_path=str(Path(self.p["lm_dir"]) / f"{lm}.bin"),
                                        unigrams=uni.read_text(encoding="utf-8").split() if uni.exists() else None,
                                        alpha=a, beta=b)
            self.beam_width = beam_width
        self.set_hotwords(hotwords or [], hotword_weight)
        self.itn = None
        if itn:
            sys.path.insert(0, str(self.p["kk_tn"]))
            from kk_tn import inverse_normalize
            self.itn = inverse_normalize
        self.reset()

    def set_hotwords(self, words, weight=10.0):
        """Contextual biasing of the final pass (pyctcdecode hotwords; needs an LM decoder): names, products, terms.
        Words are normalized like the model's output (lowercase, ё -> е). Can be changed between segments."""
        self.hotwords = [" ".join(w.lower().replace("ё", "е").split()) for w in words if w and w.strip()]
        self.hotword_weight = float(weight)

    def reset(self):
        from cache_engine import CacheEngine
        self._CE = CacheEngine
        self.vad = StreamVAD(self.p["vad"], eou=self.eou)
        self.audio = np.zeros(0, np.float32)   # stream audio from self.a0 (s) on
        self.a0 = 0.0
        self.t = 0.0                           # stream time received (s)
        self.seg0 = None                       # current segment start (s), None = no segment
        self.eng, self.lps, self.last_partial = None, [], ""
        self.prev_end = 0.0
        self.n_seg = 0

    def _slice(self, t0, t1):
        i0, i1 = int(round((t0 - self.a0) * SR)), int(round((t1 - self.a0) * SR))
        return self.audio[max(0, i0):max(0, i1)]

    def _greedy(self, lp):
        out, prev = [], -1
        for i in lp.argmax(-1).tolist():
            if i != prev and i != len(self.vocab):
                out.append(self.vocab[i])
            prev = i
        return " ".join("".join(out).split())

    def _partial_text(self, at_end=False):
        lp = torch.cat(self.lps) if self.lps else torch.zeros(0, len(self.vocab) + 1)
        words = self._greedy(lp).split()
        if words and not at_end and not self.tentative:
            words = words[:-1]
        return " ".join(words)

    @torch.inference_mode()
    def _final(self, seg_audio):
        t0 = time.perf_counter()
        os.environ["CHUNK_EVAL"], os.environ["LOOKAHEAD_EVAL"] = "0", "0"
        x = torch.from_numpy(seg_audio)[None].to(self.dev)
        if self.dev.type == "cuda":
            enc, _ = self.fm.forward(x, torch.tensor([x.shape[1]], device=self.dev))  # GigaAM autocasts fp16 itself
            with torch.autocast("cuda", torch.float16):
                lp = self.fm.head(enc)[0].float().cpu().numpy()
        else:
            feats, fl = self.fm.preprocessor(x, torch.tensor([x.shape[1]]))  # mel stays fp32
            with torch.autocast("cpu", torch.bfloat16, enabled=self.bf16_final):
                enc, _ = self.fm.encoder(feats, fl)
                lp = self.fm.head(enc)[0].float().numpy()
        t_am = time.perf_counter() - t0
        if self.dec is not None:
            kw = dict(hotwords=self.hotwords, hotword_weight=self.hotword_weight) if self.hotwords else {}
            text = self.dec.decode(lp, beam_width=self.beam_width, **kw)
        else:
            text = self._greedy(torch.from_numpy(lp))
        t_lm = time.perf_counter() - t0 - t_am
        itn = self.itn(text) if self.itn else text
        return text, itn, {"am_ms": round(1000 * t_am, 1), "lm_ms": round(1000 * t_lm, 1),
                           "total_ms": round(1000 * (time.perf_counter() - t0), 1)}

    def _start(self, t_speech):
        self.seg0 = max(t_speech - self.pad, self.prev_end, self.a0)
        self.eng, self.lps, self.last_partial = self._CE(self.cm, self.C, cpu_bf16=self.bf16_partials), [], ""
        self.fed = self.seg0

    def _feed_partial(self, upto, events):
        x = self._slice(self.fed, upto)
        self.fed = upto
        if len(x) == 0:
            return
        t0 = time.perf_counter()
        new = self.eng.feed(x)
        if new:
            self.lps += [n.float().cpu() for n in new]
            txt = self._partial_text()
            if txt != self.last_partial:
                self.last_partial = txt
                events.append({"type": "partial", "seg": self.n_seg, "t": round(upto, 3), "text": txt,
                               "compute_ms": round(1000 * (time.perf_counter() - t0), 1)})

    def _end(self, t_end_speech, events, forced=False):
        s1 = min(t_end_speech + self.pad, self.t)
        text, itn, timing = self._final(self._slice(self.seg0, s1))
        events.append({"type": "final", "seg": self.n_seg, "start": round(self.seg0, 3), "end": round(s1, 3),
                       "t_emit": round(self.t, 3), "text": text, "text_itn": itn, "forced_cut": forced, **timing})
        self.prev_end, self.seg0, self.eng = s1, None, None
        self.n_seg += 1
        # drop audio before the end of this segment (keep PAD for the next pre-roll)
        keep_from = max(self.a0, s1 - self.pad)
        self.audio = self._slice(keep_from, self.t); self.a0 = keep_from

    def feed(self, x):
        """x: float32 mono 16 kHz samples. Returns a list of partial/final events."""
        x = np.asarray(x, np.float32)
        self.audio = np.concatenate([self.audio, x]); self.t += len(x) / SR
        events = []
        for ev in self.vad.feed(x):
            if ev[0] == "start" and self.seg0 is None:
                self._start(ev[1])
            elif ev[0] == "end" and self.seg0 is not None:
                self._feed_partial(min(ev[1] + self.pad, self.t), events)
                self._end(ev[1], events)
        if self.seg0 is not None:
            self._feed_partial(self.t, events)
            if self.t - self.seg0 >= self.max_seg:
                self._end(self.t - self.pad, events, forced=True)
                self._start(self.t)  # continue the same speech in a new segment
        if self.seg0 is None and self.t - self.a0 > 2 * self.pad + 1:  # idle: keep only the pre-roll
            keep_from = self.t - self.pad
            self.audio = self._slice(keep_from, self.t); self.a0 = keep_from
        return events

    def flush(self):
        events = []
        if self.seg0 is not None:
            self._feed_partial(self.t, events)
            self._end(self.t, events)
        return events
