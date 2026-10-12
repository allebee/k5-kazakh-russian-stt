"""Cache-aware streaming engine for the chunk-trained GigaAM Conformer CTC.

Feeds audio in arbitrary blocks and emits CTC log-probs one encoder chunk (C frames x 40 ms) at a time. Each chunk is
computed once; nothing is re-decoded. It reproduces the chunk-masked full-sequence forward of streaming_patch.py
(CHUNK_EVAL=C, LOOKAHEAD_EVAL=0) frame by frame:
  mel: a frame is computed once its STFT window is complete. GigaAM-Multilingual uses center=False (no padding);
    with center=True the engine reflect-pads the start, and the end on finish(), exactly like torchaudio.
  subsampling (conv1d stride 2, x2): an output frame is computed once its input taps exist; zero padding at both ends.
    Mel + subsampling make chunk k ready 40 ms after its last sample (center=False) (the lookahead the model was trained with).
  attention: per layer, a cache of the last ceil(LEFT/C) chunks of keys and values. Keys are cached after the rotary
    rotation and the projection (GigaAM rotates before linear_k), so positions stay absolute from segment start.
  conv module: per layer, the last 2 GLU frames of the previous chunk; zeros past the chunk end (chunk-local conv).
Memory per segment: audio and mel/subsampling buffers grow with the segment (skeleton; trimming is a TODO). Keep
segments VAD-bounded: the model has only seen positions < ~30 s and pos_emb_max_len is 5000 frames (200 s).

usage (equivalence + 1-thread latency test):
  cache_engine.py OUT.json MANIFEST [MANIFEST ...]   env: C_CKPT, CHUNKS=8,16, UTTS=30, PROCS=16, MAX_S=15
"""
import copy, json, math, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace; only used by the test harness below
SR = 16000


def _rotary_fn(encoder):
    """apply_rotary_pos_emb from the module that defines the model (HF trust_remote_code) or the gigaam package."""
    fn = getattr(sys.modules.get(type(encoder).__module__), "apply_rotary_pos_emb", None)
    if fn is None:
        from gigaam.utils import apply_rotary_pos_emb as fn
    return fn


class CacheEngine:
    def __init__(self, model, chunk, left_frames=None, lookahead=0, cpu_bf16=False):
        """lookahead R > 0: honest (Emformer-style) right context. Each chunk is computed together with the next R
        frames, attention and conv limited to chunk + R; only the chunk's own keys/values/conv frames are cached, and
        the R frames are recomputed later as part of their own chunk. Latency +R frames. NOTE: this is NOT what
        streaming_patch's LOOKAHEAD_EVAL computes: its single-pass mask lets lookahead leak one chunk per layer."""
        import streaming_patch
        self.m, self.enc, self.C, self.R = model, model.encoder, chunk, lookahead
        left = streaming_patch.LEFT_FRAMES if left_frames is None else left_frames
        self.keep = max(1, -(-left // chunk)) * chunk  # cached key/value frames = whole left chunks, as in the mask
        mel = model.preprocessor.featurizer[0]
        self.mel_nc = copy.deepcopy(mel)
        self.mel_nc.spectrogram.center = False
        self.n_fft, self.hop = mel.spectrogram.n_fft, mel.spectrogram.hop_length
        self.half, self.center = self.n_fft // 2, mel.spectrogram.center
        convs = [m for m in self.enc.pre_encode.conv if isinstance(m, torch.nn.Conv1d)]
        assert self.enc.pre_encode.subsampling_type == "conv1d" and len(convs) == 2
        self.sub = convs
        self.k_sub, self.p_sub = convs[0].kernel_size[0], convs[0].padding[0]
        dw = self.enc.layers[0].conv.depthwise_conv
        self.dw_left = (dw.kernel_size[0] - 1) // 2
        # encoder frame m needs mel frame 4m + 3k - 3p - 3, whose window ends at i*hop + n_fft (- n_fft/2 if centred)
        win_end = self.n_fft - (self.half if self.center else 0)
        self.lookahead_ms = 1000 * ((3 * self.k_sub - 3 * self.p_sub - 3 - 4) * self.hop + win_end) / SR
        self.dev = next(model.parameters()).device
        # on GPU, run like GigaAM's own forward: mel in fp32, encoder (incl. subsampling) and head under fp16 autocast;
        # on CPU optionally bf16 autocast (AMX on Sapphire/Emerald Rapids)
        self.amp = self.dev.type == "cuda"
        self.ac = ("cuda", torch.float16, True) if self.amp else ("cpu", torch.bfloat16, bool(cpu_bf16))
        if not hasattr(self.enc.pos_enc, "pe"):
            self.enc.pos_enc.extend_pe(self.enc.pos_emb_max_len, self.dev)
        self.reset()

    def reset(self):
        dev = self.dev
        self.audio = torch.zeros(0, device=dev)
        self.n_mel = 0                                        # mel frames computed
        z = torch.zeros(self.sub[0].in_channels, self.p_sub, device=dev)
        self.mel_buf = z.clone()                              # (64, p + n_mel): left zero padding + mel frames
        self.s1_buf = torch.zeros(self.sub[0].out_channels, self.p_sub, device=dev)  # stage-1 frames, left zero padded
        self.n_s1 = self.n_s2 = 0
        self.frames = torch.zeros(0, self.enc.layers[0].norm_out.normalized_shape[0], device=dev)  # subsampled, not yet encoded
        self.pos = 0                                          # encoder frames already encoded
        D = self.frames.shape[1]
        a = self.enc.layers[0].self_attn
        self.kc = [torch.zeros(1, a.h, 0, a.d_k, device=dev) for _ in self.enc.layers]
        self.vc = [torch.zeros(1, a.h, 0, a.d_k, device=dev) for _ in self.enc.layers]
        self.cc = [torch.zeros(1, D, self.dw_left, device=dev) for _ in self.enc.layers]

    # ---- front end -------------------------------------------------------------------------------------------
    def _mel(self, padded, i0, i1):
        """log-mel frames i0..i1-1 (center coordinates) from audio that is already left reflect-padded."""
        seg = padded[i0 * self.hop:(i1 - 1) * self.hop + self.n_fft]
        return torch.log(self.mel_nc(seg).clamp_(1e-9, 1e9))

    def _subsample(self, final):
        """Run both strided convs over whatever new input frames are complete; append to self.frames."""
        k, p = self.k_sub, self.p_sub
        for stage, (conv, src, attr) in enumerate([(self.sub[0], "mel_buf", "n_s1"), (self.sub[1], "s1_buf", "n_s2")]):
            buf = getattr(self, src)
            if final:
                buf = torch.cat([buf, torch.zeros(buf.shape[0], p, device=self.dev)], 1)
            done = getattr(self, attr)
            ready = (buf.shape[1] - k) // 2 + 1 if buf.shape[1] >= k else 0
            if ready > done:
                x = buf[:, 2 * done:2 * (ready - 1) + k]
                with torch.autocast(self.ac[0], self.ac[1], enabled=self.ac[2]):
                    y = torch.relu(F.conv1d(x[None], conv.weight, conv.bias, stride=2))[0]
                if stage == 0:
                    self.s1_buf = torch.cat([self.s1_buf, y], 1)
                else:
                    self.frames = torch.cat([self.frames, y.T], 0)
                setattr(self, attr, ready)

    def _front(self, final=False):
        N = self.audio.shape[0]
        if self.center:  # torchaudio center=True: reflect padding of n_fft/2 at both ends
            if N <= self.half:
                return
            total = N // self.hop + 1 if final else (N - self.half) // self.hop + 1  # frames with a complete window
        else:            # center=False (GigaAM-Multilingual): no padding, frame i = audio[i*hop : i*hop + n_fft]
            if N < self.n_fft:
                return
            total = (N - self.n_fft) // self.hop + 1
        if total > self.n_mel:
            if self.center:
                padded = F.pad(self.audio[None, None], (self.half, self.half if final else 0), mode="reflect")[0, 0]
            else:
                padded = self.audio
            m = self._mel(padded, self.n_mel, total)
            self.mel_buf = torch.cat([self.mel_buf, m], 1)
            self.n_mel = total
        self._subsample(final)

    # ---- encoder ---------------------------------------------------------------------------------------------
    def _encode(self, x, keep=None):
        """x: (n, D) subsampled frames at positions self.pos.. -> log-probs of the first `keep` frames (default n).
        Caches keys/values/conv inputs of those `keep` frames only (the rest is right context)."""
        keep = x.shape[0] if keep is None else keep
        apply_rotary_pos_emb = _rotary_fn(self.enc)
        n, D = x.shape
        pe = self.enc.pos_enc.pe
        half = pe.shape[0] // 2
        cos, sin = pe[self.pos:self.pos + n], pe[half + self.pos:half + self.pos + n]
        h = x[None]
        for l, L in enumerate(self.enc.layers):
            res = h
            res = res + L.fc_factor * L.feed_forward1(L.norm_feed_forward1(res))
            a, y = L.self_attn, L.norm_self_att(res)
            yt = y.transpose(0, 1).reshape(n, 1, a.h, a.d_k)
            qr, kr = apply_rotary_pos_emb(yt, yt, cos, sin)
            q = a.linear_q(qr.reshape(n, 1, D).transpose(0, 1)).view(1, n, a.h, a.d_k).transpose(1, 2)
            k = a.linear_k(kr.reshape(n, 1, D).transpose(0, 1)).view(1, n, a.h, a.d_k).transpose(1, 2)
            v = a.linear_v(y).view(1, n, a.h, a.d_k).transpose(1, 2)
            K, V = torch.cat([self.kc[l], k], 2), torch.cat([self.vc[l], v], 2)
            o = F.scaled_dot_product_attention(q, K, V)
            res = res + a.linear_out(o.transpose(1, 2).reshape(1, n, D))
            cv = L.conv
            g = F.glu(cv.pointwise_conv1(L.norm_conv(res).transpose(1, 2)), dim=1)  # (1, D, n)
            xin = torch.cat([self.cc[l], g, torch.zeros(1, D, self.dw_left, device=self.dev, dtype=g.dtype)], 2)
            z = F.conv1d(xin, cv.depthwise_conv.weight, cv.depthwise_conv.bias, groups=D)
            z = cv.batch_norm(z) if cv.norm_type == "batch_norm" else cv.batch_norm(z.transpose(1, 2)).transpose(1, 2)
            res = res + cv.pointwise_conv2(cv.activation(z)).transpose(1, 2)
            res = res + L.fc_factor * L.feed_forward2(L.norm_feed_forward2(res))
            h = L.norm_out(res)
            n_old = self.kc[l].shape[2]
            self.kc[l], self.vc[l] = K[:, :, :n_old + keep][:, :, -self.keep:], V[:, :, :n_old + keep][:, :, -self.keep:]
            self.cc[l] = torch.cat([self.cc[l], g[:, :, :keep]], 2)[:, :, -self.dw_left:]
        self.pos += keep
        return self.m.head(h[:, :keep].transpose(1, 2))[0]

    @torch.inference_mode()
    def feed(self, samples):
        """Add audio; return log-probs of every chunk that became complete (list of (C, V) tensors)."""
        self.audio = torch.cat([self.audio, torch.as_tensor(samples, dtype=torch.float32).to(self.dev)])
        self._front()
        out = []
        while self.frames.shape[0] >= self.C + self.R:
            with torch.autocast(self.ac[0], self.ac[1], enabled=self.ac[2]):
                out.append(self._encode(self.frames[:self.C + self.R], self.C).float())
            self.frames = self.frames[self.C:]
        return out

    @torch.inference_mode()
    def finish(self):
        """End of segment: pad like the offline forward and flush the last (possibly partial) chunk(s)."""
        self._front(final=True)
        out = []
        while self.frames.shape[0]:
            with torch.autocast(self.ac[0], self.ac[1], enabled=self.ac[2]):
                out.append(self._encode(self.frames[:self.C + self.R], min(self.C, self.frames.shape[0])).float())
            self.frames = self.frames[self.C:]
        return out


# ---- equivalence + latency test ------------------------------------------------------------------------------
M = {}


def init():
    import gigaam, streaming_patch
    torch.set_num_threads(1)
    gigaam._CACHE_DIR = str(STT / ".gigaam-cache")
    streaming_patch.enable()
    M["m"] = gigaam.load_model(os.environ.get("C_CKPT", str(STT / "runs/stream_multilingual_large_ctc/final_stream.ckpt")),
                               fp16_encoder=False, device="cpu").eval()
    M["vocab"] = [str(v) for v in M["m"].decoding.tokenizer.vocab]
    M["lookahead_ms"] = CacheEngine(M["m"], 8).lookahead_ms


def greedy(lp):
    blank, out, prev = len(M["vocab"]), [], -1
    for i in lp.argmax(-1).tolist():
        if i != prev and i != blank:
            out.append(M["vocab"][i])
        prev = i
    return " ".join("".join(out).split())


def offline(wav, C):
    os.environ["CHUNK_EVAL"], os.environ["LOOKAHEAD_EVAL"] = str(C), "0"
    x = torch.from_numpy(wav)[None]
    with torch.inference_mode():
        enc, _ = M["m"].forward(x, torch.tensor([x.shape[1]]))
        return M["m"].head(enc)[0]


def run_utt(job):
    wav, C = job
    ref_lp = offline(wav, C)                       # = the re-decode engine's last partial / chunk-masked decode
    eng = CacheEngine(M["m"], C)
    block = 1600                                   # 100 ms blocks: deliberately not aligned to the 40*C ms chunks
    chunks, t_chunk = [], []
    for i in range(0, len(wav), block):
        t0 = time.perf_counter()
        new = eng.feed(wav[i:i + block])
        dt = time.perf_counter() - t0
        if new:
            t_chunk += [dt / len(new)] * len(new)  # a block completes at most one chunk at C >= 4
        chunks += new
    t0 = time.perf_counter()
    tail = eng.finish()
    t_finish = time.perf_counter() - t0
    lp = torch.cat(chunks + tail)
    # prefix recompute partials (today's re-decode) at each chunk boundary vs the engine's committed chunks
    agree = total = 0
    for k in range(1, len(chunks) + 1):
        n = k * C * 640                            # samples covering chunks 0..k-1
        if n > len(wav):
            break
        pre = offline(wav[:n], C)[:k * C]
        a, b = greedy(pre).split(), greedy(torch.cat(chunks[:k])).split()
        total += 1
        agree += a[:-1] == b[:-1]                  # the sim hides the last (still growing) word
    same_len = lp.shape[0] == ref_lp.shape[0]
    d = (lp - ref_lp).abs() if same_len else None
    return {"C": C, "frames": int(lp.shape[0]), "ref_frames": int(ref_lp.shape[0]),
            "max_abs_dlogp": float(d.max()) if same_len else None,
            "argmax_agree": float((lp.argmax(-1) == ref_lp.argmax(-1)).float().mean()) if same_len else None,
            "hyp_engine": greedy(lp), "hyp_redecode": greedy(ref_lp),
            "partials_agree": agree, "partials_total": total,
            "chunk_compute_s": t_chunk, "finish_s": t_finish, "audio_s": len(wav) / SR}


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))  # textnorm
    import jiwer, soundfile as sf
    from textnorm import normalize
    out, manifests = sys.argv[1], sys.argv[2:]
    n_utt, max_s = int(os.environ.get("UTTS", "30")), float(os.environ.get("MAX_S", "15"))
    jobs, refs = [], []
    per = -(-n_utt // len(manifests))
    for m in manifests:
        rows = [json.loads(l) for l in open(m, encoding="utf-8")]
        random.Random(2).shuffle(rows)
        k = 0
        for r in rows:
            w, sr = sf.read(r["path"], dtype="float32")
            w = w if w.ndim == 1 else w.mean(1)
            if sr != SR or len(w) / SR > max_s:
                continue
            jobs.append(w); refs.append(normalize(r["ref"])); k += 1
            if k == per:
                break
    chunks = [int(c) for c in os.environ.get("CHUNKS", "8,16").split(",")]
    with ProcessPoolExecutor(int(os.environ.get("PROCS", "16")), initializer=init) as ex:
        res = list(ex.map(run_utt, [(w, C) for C in chunks for w in jobs]))
    init()  # parent: lookahead_ms, and a clean latency pass (one process, 1 thread, no parallel prefix recompute)
    clean = {}
    for C in chunks:
        ts = []
        for w in jobs[:int(os.environ.get("LAT_UTTS", "6"))]:
            eng = CacheEngine(M["m"], C)
            for i in range(0, len(w), 1600):
                t0 = time.perf_counter(); new = eng.feed(w[i:i + 1600]); dt = time.perf_counter() - t0
                ts += [dt / len(new)] * len(new) if new else []
        clean[C] = ts
    summary = {}
    for C in chunks:
        rs = [r for r in res if r["C"] == C]
        tc = [t for r in rs for t in r["chunk_compute_s"]]
        summary[f"{C * 40}ms"] = {
            "utts": len(rs), "audio_min": round(sum(r["audio_s"] for r in rs) / 60, 1),
            "frame_count_match": sum(r["frames"] == r["ref_frames"] for r in rs),
            "max_abs_dlogp": max(r["max_abs_dlogp"] for r in rs if r["max_abs_dlogp"] is not None),
            "argmax_agree_min": min(r["argmax_agree"] for r in rs if r["argmax_agree"] is not None),
            "text_identical": sum(r["hyp_engine"] == r["hyp_redecode"] for r in rs),
            "wer_engine": round(100 * jiwer.wer(refs, [normalize(r["hyp_engine"]) for r in rs]), 2),
            "wer_redecode": round(100 * jiwer.wer(refs, [normalize(r["hyp_redecode"]) for r in rs]), 2),
            "partials_identical_vs_prefix_recompute": f'{sum(r["partials_agree"] for r in rs)}/{sum(r["partials_total"] for r in rs)}',
            "chunk_compute_ms_cpu1thread_16procs_busy": {"mean": round(1000 * float(np.mean(tc)), 1), "p90": round(1000 * float(np.percentile(tc, 90)), 1),
                                            "max": round(1000 * float(np.max(tc)), 1)},
            "rtf_cpu1thread": round(float(np.mean(tc)) / (C * 0.04), 3),
            "clean_single_process_chunk_ms": {"mean": round(1000 * float(np.mean(clean[C])), 1),
                                              "p90": round(1000 * float(np.percentile(clean[C], 90)), 1)},
            "clean_rtf": round(float(np.mean(clean[C])) / (C * 0.04), 3),
            "finish_ms_mean": round(1000 * float(np.mean([r["finish_s"] for r in rs])), 1),
            "algorithmic_lookahead_ms": M["lookahead_ms"],
        }
    json.dump({"summary": summary, "per_utt": [{k: v for k, v in r.items() if k != "chunk_compute_s"} for r in res]},
              open(out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1))
