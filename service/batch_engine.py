"""Batched cache-aware streaming engine: all active streams' ready chunks go through ONE encoder forward per tick.

Per-layer caches live in fixed-shape slot pools (max_streams x heads x KEEP x d_k for keys/values, max_streams x D x 2
for the conv module); a tick gathers the active slots, runs the 16 layers once on a (B, C, D) batch, and writes the
shifted caches back with one indexed assignment per layer. Partly filled caches (stream start) are handled by a key
padding mask; rotary positions are gathered per stream. Front ends (mel + subsampling) stay per stream, reusing
CacheEngine._front. Semantics are identical to CacheEngine (R = 0): same keys/values/conv taps per chunk.
Finals: final_batch() runs the full-context model on several padded segments in one forward.

usage (CPU equivalence test vs the per-stream engine): batch_engine.py OUT.json   env: C_CKPT, N=6, CHUNK=8
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

STT = Path(os.environ.get("K5_STT_ROOT", ".")).resolve()  # workspace; only used by the test harness below
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_engine import CacheEngine


class BatchedEngine:
    def __init__(self, model, chunk, max_streams=128, cpu_bf16=False):
        from cache_engine import _rotary_fn
        self.rot = _rotary_fn(model.encoder)
        self.m, self.enc, self.C = model, model.encoder, chunk
        self.dev = next(model.parameters()).device
        self.proto = CacheEngine(model, chunk)            # constants (keep, dw_left, ...) + pe buffer
        self.keep, self.dwl = self.proto.keep, self.proto.dw_left
        a = self.enc.layers[0].self_attn
        self.H, self.dk = a.h, a.d_k
        self.D = self.H * self.dk
        S, L = max_streams, len(self.enc.layers)
        z = lambda *s: torch.zeros(*s, device=self.dev)
        self.Kp = [z(S, self.H, self.keep, self.dk) for _ in range(L)]
        self.Vp = [z(S, self.H, self.keep, self.dk) for _ in range(L)]
        self.Cp = [z(S, self.D, self.dwl) for _ in range(L)]
        self.valid = torch.zeros(S, dtype=torch.long, device=self.dev)
        self.pos = torch.zeros(S, dtype=torch.long, device=self.dev)
        self.front = {}                                    # slot -> CacheEngine used only for its front end
        self.free = list(range(S))
        amp = self.dev.type == "cuda"
        self.ac = ("cuda", torch.float16, True) if amp else ("cpu", torch.bfloat16, bool(cpu_bf16))

    # ---- stream management ---------------------------------------------------------------------------------
    def open(self):
        s = self.free.pop(0)
        self.front[s] = CacheEngine(self.m, self.C)
        self.valid[s] = 0; self.pos[s] = 0
        for l in range(len(self.Kp)):
            self.Cp[l][s].zero_()
        return s

    def close(self, s):
        self.front.pop(s, None); self.free.append(s)

    @torch.inference_mode()
    def feed(self, s, samples):
        """Front end only (mel + subsampling); encoding happens in step()."""
        fe = self.front[s]
        fe.audio = torch.cat([fe.audio, torch.as_tensor(samples, dtype=torch.float32).to(self.dev)])
        fe._front()

    @torch.inference_mode()
    def finish(self, s):
        """End of stream s: flush the front end and encode the remaining (< C) frames on their own."""
        fe = self.front[s]
        fe._front(final=True)
        out = []
        while fe.frames.shape[0]:
            n = min(self.C, fe.frames.shape[0])
            out.append(self._encode([s], fe.frames[:n][None], n)[0])
            fe.frames = fe.frames[n:]
        return out

    @torch.inference_mode()
    def step(self):
        """One tick: every stream with a full chunk ready is encoded in one batched forward. -> {slot: (C, V)}"""
        ready = [s for s, fe in self.front.items() if fe.frames.shape[0] >= self.C]
        if not ready:
            return {}
        x = torch.stack([self.front[s].frames[:self.C] for s in ready])
        lp = self._encode(ready, x, self.C)
        for s in ready:
            self.front[s].frames = self.front[s].frames[self.C:]
        return {s: lp[i] for i, s in enumerate(ready)}

    # ---- batched encoder -------------------------------------------------------------------------------------
    def _encode(self, slots, x, n):
        with torch.autocast(self.ac[0], self.ac[1], enabled=self.ac[2]):
            return self._encode_inner(slots, x, n).float()

    def _encode_inner(self, slots, x, n):
        B, H, dk, D, keep, dwl = len(slots), self.H, self.dk, self.D, self.keep, self.dwl
        sl = torch.tensor(slots, device=self.dev)
        pe = self.enc.pos_enc.pe
        half = pe.shape[0] // 2
        idx = (self.pos[sl][:, None] + torch.arange(n, device=self.dev)[None, :]).T      # (n, B)
        cos, sin = pe[idx].squeeze(2), pe[half + idx].squeeze(2)                          # (n, B, 1, dk)
        valid = self.valid[sl]
        cache_ok = torch.arange(keep, device=self.dev)[None, :] >= (keep - valid)[:, None]  # (B, keep) left padding
        mask = torch.cat([cache_ok, torch.ones(B, n, dtype=torch.bool, device=self.dev)], 1)[:, None, None, :]
        h = x
        for l, L in enumerate(self.enc.layers):
            res = h
            res = res + L.fc_factor * L.feed_forward1(L.norm_feed_forward1(res))
            a, y = L.self_attn, L.norm_self_att(res)
            yt = y.transpose(0, 1).reshape(n, B, H, dk)
            qr, kr = self.rot(yt, yt, cos, sin)
            q = a.linear_q(qr.reshape(n, B, D).transpose(0, 1)).view(B, n, H, dk).transpose(1, 2)
            k = a.linear_k(kr.reshape(n, B, D).transpose(0, 1)).view(B, n, H, dk).transpose(1, 2)
            v = a.linear_v(y).view(B, n, H, dk).transpose(1, 2)
            Kc, Vc = self.Kp[l][sl], self.Vp[l][sl]
            K, V = torch.cat([Kc.to(k.dtype), k], 2), torch.cat([Vc.to(v.dtype), v], 2)
            o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
            res = res + a.linear_out(o.transpose(1, 2).reshape(B, n, D))
            cv = L.conv
            g = F.glu(cv.pointwise_conv1(L.norm_conv(res).transpose(1, 2)), dim=1)      # (B, D, n)
            cc = self.Cp[l][sl].to(g.dtype)
            xin = torch.cat([cc, g, torch.zeros(B, D, dwl, device=self.dev, dtype=g.dtype)], 2)
            zz = F.conv1d(xin, cv.depthwise_conv.weight, cv.depthwise_conv.bias, groups=D)
            zz = cv.batch_norm(zz) if cv.norm_type == "batch_norm" else cv.batch_norm(zz.transpose(1, 2)).transpose(1, 2)
            res = res + cv.pointwise_conv2(cv.activation(zz)).transpose(1, 2)
            res = res + L.fc_factor * L.feed_forward2(L.norm_feed_forward2(res))
            h = L.norm_out(res)
            # shift the caches by n frames and append this chunk's keys/values/conv inputs (one write per pool)
            self.Kp[l][sl] = torch.cat([Kc, k.float()], 2)[:, :, -keep:]
            self.Vp[l][sl] = torch.cat([Vc, v.float()], 2)[:, :, -keep:]
            self.Cp[l][sl] = torch.cat([self.Cp[l][sl], g.float()], 2)[:, :, -dwl:]
        self.valid[sl] = torch.clamp(valid + n, max=keep)
        self.pos[sl] = self.pos[sl] + n
        return self.m.head(h.transpose(1, 2))


@torch.inference_mode()
def final_batch(model, segs, cpu_bf16=True):
    """Full-context decode of several segments in one padded forward -> list of (T_i, V) log-probs."""
    dev = next(model.parameters()).device
    os.environ["CHUNK_EVAL"], os.environ["LOOKAHEAD_EVAL"] = "0", "0"
    lens = torch.tensor([len(s) for s in segs], device=dev)
    x = torch.zeros(len(segs), int(lens.max()), device=dev)
    for i, s in enumerate(segs):
        x[i, :len(s)] = torch.as_tensor(s, device=dev)
    feats, fl = model.preprocessor(x, lens)
    on_gpu = dev.type == "cuda"
    with torch.autocast("cuda" if on_gpu else "cpu", torch.float16 if on_gpu else torch.bfloat16, enabled=on_gpu or cpu_bf16):
        enc, el = model.encoder(feats, fl)
        lp = model.head(enc).float()
    return [lp[i, :int(el[i])] for i in range(len(segs))]


# ---- CPU equivalence test vs the per-stream engine ---------------------------------------------------------------
if __name__ == "__main__":
    import random, gigaam, streaming_patch, soundfile as sf
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    gigaam._CACHE_DIR = str(STT / ".gigaam-cache")
    streaming_patch.enable()
    cm = gigaam.load_model(os.environ.get("C_CKPT", str(STT / "runs/stream_multilingual_large_ctc/final_stream.ckpt")),
                           fp16_encoder=False, device="cpu").eval()
    fm = gigaam.load_model(str(STT / "runs/full_multilingual_large_ctc/final_a1.00.ckpt"), fp16_encoder=False, device="cpu").eval()
    fm.encoder._force_full = True
    vocab = [str(v) for v in cm.decoding.tokenizer.vocab]

    def greedy(lp):
        out, prev = [], -1
        for i in lp.argmax(-1).tolist():
            if i != prev and i != len(vocab):
                out.append(vocab[i])
            prev = i
        return " ".join("".join(out).split())

    N, C = int(os.environ.get("N", "6")), int(os.environ.get("CHUNK", "8"))
    rows = [json.loads(l) for s in ("dev_fleurs_kk", "dev_fleurs_ru", "dev_ksc2_spont")
            for l in open(STT / f"eval/manifests/{s}.jsonl", encoding="utf-8")]
    random.Random(9).shuffle(rows)
    wavs = []
    for r in rows:
        w, sr = sf.read(r["path"], dtype="float32")
        if sr == 16000 and w.ndim == 1 and len(w) < 15 * 16000:
            wavs.append(w)
        if len(wavs) == N:
            break
    # reference: per-stream engine
    t0 = time.perf_counter()
    ref = []
    for w in wavs:
        e, lps = CacheEngine(cm, C), []
        for i in range(0, len(w), 1600):
            lps += e.feed(w[i:i + 1600])
        ref.append(torch.cat(lps + e.finish()))
    t_ref = time.perf_counter() - t0
    # batched: one chunk of audio per tick per stream (so every active stream has a chunk ready each tick), stream i
    # joins at tick i -> each batch mixes streams at different positions and cache fill levels
    be = BatchedEngine(cm, C, max_streams=16)
    step_n = C * 640
    slots, outs, started, done = {}, {i: [] for i in range(N)}, set(), set()
    t0 = time.perf_counter(); tick = 0; batch_sizes = []
    while len(done) < N:
        for i in range(N):
            if i <= tick and i not in started:
                slots[i] = be.open(); started.add(i)
            if i in started and i not in done:
                off = (tick - i) * step_n
                if off < len(wavs[i]):
                    be.feed(slots[i], wavs[i][off:off + step_n])
        res = be.step()
        batch_sizes.append(len(res))
        for i in started - done:
            if slots[i] in res:
                outs[i].append(res[slots[i]])
        for i in list(started - done):
            if (tick - i + 1) * step_n >= len(wavs[i]) and be.front[slots[i]].frames.shape[0] < C:
                outs[i] += be.finish(slots[i]); be.close(slots[i]); done.add(i)
        tick += 1
    t_bat = time.perf_counter() - t0
    diffs, same = [], 0
    for i in range(N):
        b = torch.cat(outs[i])
        if b.shape == ref[i].shape:
            diffs.append(float((b - ref[i]).abs().max()))
        same += greedy(b) == greedy(ref[i])
    # finals: batched vs one by one
    singles = [final_batch(fm, [w])[0] for w in wavs]
    batched = final_batch(fm, wavs)
    fin_same = sum(greedy(a) == greedy(b) for a, b in zip(singles, batched))
    fin_diff = max(float((a - b).abs().max()) for a, b in zip(singles, batched))
    fin_argmax = min(float((a.argmax(-1) == b.argmax(-1)).float().mean()) for a, b in zip(singles, batched))
    res = {"N": N, "chunk_ms": C * 40, "text_identical": f"{same}/{N}", "frame_shapes_equal": len(diffs),
           "max_abs_dlogp": max(diffs) if diffs else None, "max_batch": max(batch_sizes),
           "cpu_seconds_per_stream_engine": round(t_ref, 2), "cpu_seconds_batched": round(t_bat, 2),
           "finals_batched_text_identical": f"{fin_same}/{N}", "finals_batched_max_abs_dlogp": round(fin_diff, 6),
           "finals_batched_argmax_agreement_min": round(fin_argmax, 4), "mean_batch": round(float(np.mean([b for b in batch_sizes if b])), 2)}
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(res, indent=1))
