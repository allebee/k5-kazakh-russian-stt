"""Chunk-aware (streaming) mode for GigaAM's Conformer encoder, as a monkeypatch.

Attention: a frame sees its own chunk, R lookahead frames past the chunk end, and previous chunks up to LEFT_FRAMES
  of history. Rotary positions stay absolute, so a cache-aware engine can store already-rotated keys.
Conv module (depthwise, kernel 5, symmetric): chunk-local. Output frames keep their 2 frames of left context, but
  taps reading beyond chunk_end + R are removed exactly (cheap subtraction, no recompute). With R >= 2 nothing is removed.
Subsampling (conv1d x4) is untouched: a streaming engine gives each chunk a few mel frames (~30 ms) of lookahead.

Chunk size C and lookahead R, in encoder frames (40 ms):
  training: C from TRAIN_CHUNKS (full context with probability P_FULL), R from TRAIN_LOOKAHEADS, per forward pass
  eval:     CHUNK_EVAL (0 = full context), LOOKAHEAD_EVAL
  an encoder with `_force_full = True` (e.g. a frozen KD teacher) always runs full context
Enable with: import streaming_patch; streaming_patch.enable()
"""
import os
import random

import torch

LEFT_FRAMES = int(os.environ.get("CHUNK_LEFT_FRAMES", "100"))  # 4 s of attention history
TRAIN_CHUNKS = [int(c) for c in os.environ.get("TRAIN_CHUNKS", "4,8,16,32").split(",")]  # 160/320/640/1280 ms
TRAIN_LOOKAHEADS = [int(r) for r in os.environ.get("TRAIN_LOOKAHEADS", "0").split(",")]
P_FULL = float(os.environ.get("P_FULL", "0.15"))


def _pick(training):
    if training:
        return (0, 0) if random.random() < P_FULL else (random.choice(TRAIN_CHUNKS), random.choice(TRAIN_LOOKAHEADS))
    return int(os.environ.get("CHUNK_EVAL", "0")), int(os.environ.get("LOOKAHEAD_EVAL", "0"))


def chunk_att_mask(T, C, device, R=0):
    """(T, T) bool, True = masked: key beyond the query's chunk end + R, or more than LEFT_FRAMES of history behind."""
    idx = torch.arange(T, device=device)
    ch = idx // C
    left_chunks = max(1, -(-LEFT_FRAMES // C))
    chunk_end = (ch + 1) * C - 1
    return (idx[None, :] > chunk_end[:, None] + R) | (ch[None, :] < ch[:, None] - left_chunks)


def enable(module=None):
    """Patch the Conformer classes of `module` (the HF trust_remote_code module of a loaded model:
    sys.modules[type(model.encoder).__module__]) or, by default, of the gigaam package. Idempotent per module."""
    if module is None:
        from gigaam import encoder as E
    else:
        E = module
    if getattr(E, "_streaming_patched", False):
        return
    E._streaming_patched = True

    orig_encoder_forward = E.ConformerEncoder.forward
    orig_conv_forward = E.ConformerConvolution.forward
    orig_attn_forward = E.RotaryPositionMultiHeadAttention.forward

    def conv_forward(self, x, pad_mask=None):
        C, R = getattr(self, "_chunk", (0, 0))
        if not C or R >= 2:
            return orig_conv_forward(self, x, pad_mask)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = torch.nn.functional.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        y = self.depthwise_conv(x)  # (B, D, T), symmetric kernel 5: taps -2..+2
        if self.depthwise_conv.kernel_size[0] == 5:
            T = x.shape[-1]
            w = self.depthwise_conv.weight[:, 0, :]
            x1 = torch.nn.functional.pad(x[..., 1:], (0, 1))  # x[t+1]
            x2 = torch.nn.functional.pad(x[..., 2:], (0, 2))  # x[t+2]
            # distance to the last visible frame (chunk end + R): 0 -> drop taps +1,+2; 1 -> drop tap +2
            dist = (C - 1 - torch.arange(T, device=x.device) % C) + R
            last = (dist == 0).to(x.dtype)
            second = (dist == 1).to(x.dtype)
            y = y - last * (w[:, 3, None] * x1 + w[:, 4, None] * x2) - second * (w[:, 4, None] * x2)
        if self.norm_type == "batch_norm":
            y = self.batch_norm(y)
        else:
            y = self.batch_norm(y.transpose(1, 2)).transpose(1, 2)
        y = self.activation(y)
        y = self.pointwise_conv2(y)
        return y.transpose(1, 2)

    def encoder_forward(self, audio_signal, length):
        cr = (0, 0) if getattr(self, "_force_full", False) else _pick(self.training)
        for layer in self.layers:
            layer.conv._chunk = cr
            layer.self_attn._chunk = cr
        self._chunk = cr
        return orig_encoder_forward(self, audio_signal, length)

    def attn_forward(self, query, key, value, pos_emb, mask=None):
        C, R = getattr(self, "_chunk", (0, 0))
        if C:
            B, T = query.shape[0], query.shape[1]
            cm = chunk_att_mask(T, C, query.device, R).unsqueeze(0)
            mask = cm.expand(B, T, T) if mask is None else (mask | cm)
        return orig_attn_forward(self, query, key, value, pos_emb, mask=mask)

    E.ConformerEncoder.forward = encoder_forward
    E.ConformerConvolution.forward = conv_forward
    E.RotaryPositionMultiHeadAttention.forward = attn_forward
    print(f"[streaming_patch] chunk-aware mode on: train chunks {TRAIN_CHUNKS} frames, lookaheads {TRAIN_LOOKAHEADS}, "
          f"P(full)={P_FULL}, left {LEFT_FRAMES} frames", flush=True)
