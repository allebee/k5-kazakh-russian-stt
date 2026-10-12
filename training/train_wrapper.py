"""Run GigaAM's official train_utils/train.py with our additions: speed perturbation, online reverb + noise,
duration-bucketed batches, EMA, throughput logging, a train-mode fix, and optional chunk-aware (streaming) training
with knowledge distillation.

Paths (env): K5_STT_ROOT = workspace (default: current directory; GigaAM's model cache goes to
$K5_STT_ROOT/.gigaam-cache), GIGAAM_DIR = a clone of github.com/salute-developers/GigaAM (default
$K5_STT_ROOT/GigaAM), NOISE_DIR = RIRS_NOISES + MUSAN (default $K5_STT_ROOT/data/noise). GigaAM's trainer changes
directory, so pass absolute manifest paths.
usage: train_wrapper.py <train.py args...>   (MEM_FRACTION env: optional per-process CUDA memory cap, e.g. 0.75)
"""
import os, runpy, sys

import torch

STT = os.path.abspath(os.environ.get("K5_STT_ROOT", "."))
GIGAAM_DIR = os.path.abspath(os.environ.get("GIGAAM_DIR", f"{STT}/GigaAM"))
frac = float(os.environ.get("MEM_FRACTION", "1.0"))
if torch.cuda.is_available() and frac < 1.0:
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"[wrapper] CUDA memory cap: {frac:.2f} x {total:.1f} GiB = {frac * total:.1f} GiB", flush=True)

import random

import torchaudio.functional as AF

import gigaam
from gigaam import utils as gutils

gigaam._CACHE_DIR = f"{STT}/.gigaam-cache"

# Speed perturbation 0.9 / 1.0 / 1.1 (Kaldi-style: tempo and pitch), training manifests only (file name train*).
# GigaAM's AudioDataset has none. Resampling 11->10 plays 1.1x faster; 9->10 plays 0.9x.
SPEED = os.environ.get("SPEED_PERTURB", "1") == "1"
_init, _load = gutils.AudioDataset.__init__, gutils.AudioDataset._load_audio


# Online room reverb + additive noise: RIR p=0.3 from OpenSLR 28 simulated RIRs
# (Apache 2.0), noise p=0.3 at 5-25 dB SNR from OpenSLR 28 point-source noises (+ MUSAN noise once it has landed).
# Not applied to the offline telephony copies (data/aug/tel_*): there the channel already came after the voice.
NOISE_DIR = os.environ.get("NOISE_DIR", f"{STT}/data/noise")
P_RIR, P_NOISE = float(os.environ.get("P_RIR", "0.3")), float(os.environ.get("P_NOISE", "0.3"))
RIRS, NOISES = [], []


def _scan_noise():
    import glob
    RIRS[:] = glob.glob(f"{NOISE_DIR}/RIRS_NOISES/simulated_rirs/*/*/*.wav")
    NOISES[:] = glob.glob(f"{NOISE_DIR}/RIRS_NOISES/pointsource_noises/*.wav") + glob.glob(f"{NOISE_DIR}/musan/noise/*/*.wav")
    print(f"[wrapper] online aug: {len(RIRS)} RIRs (p={P_RIR}), {len(NOISES)} noises (p={P_NOISE})", flush=True)


def _read_mono(path):
    import soundfile as sf
    x, _ = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(x[:, 0].copy())


def _reverb_noise(wav):
    if RIRS and random.random() < P_RIR:
        rir = _read_mono(random.choice(RIRS))
        rir = rir / (rir.norm() + 1e-8)
        peak = int(rir.abs().argmax())
        y = AF.fftconvolve(wav, rir)[peak: peak + wav.shape[-1]]
        wav = y * (wav.abs().max() / (y.abs().max() + 1e-8))
    if NOISES and random.random() < P_NOISE:
        n = _read_mono(random.choice(NOISES))
        if len(n) < len(wav):
            n = n.repeat(len(wav) // max(len(n), 1) + 1)
        off = random.randint(0, len(n) - len(wav))
        n = n[off: off + len(wav)]
        snr = random.uniform(5, 25)
        wav = wav + n * torch.sqrt((wav.pow(2).mean() + 1e-10) / ((n.pow(2).mean() + 1e-10) * 10 ** (snr / 10)))
    return wav


def _patched_init(self, data, *args, **kwargs):
    _init(self, data, *args, **kwargs)
    self._train = isinstance(data, str) and os.path.basename(data).startswith("train")
    if self._train:
        _scan_noise()  # once, in the main process, before DataLoader workers fork


def _read_audio(item):
    if not isinstance(item, str):
        return _load(item)  # original is a staticmethod
    import soundfile as sf
    x, sr = sf.read(item, dtype="float32", always_2d=True)
    wav = torch.from_numpy(x.mean(1) if x.shape[1] > 1 else x[:, 0].copy())
    return AF.resample(wav, sr, 16000) if sr != 16000 else wav


def _patched_load(self, item):
    wav = _read_audio(item)
    if getattr(self, "_train", False):
        if SPEED:
            f = random.choice((0.9, 1.0, 1.1))
            if f != 1.0:
                wav = AF.resample(wav, orig_freq=round(f * 10), new_freq=10)
        if isinstance(item, str) and "/data/aug/tel_" not in item:
            wav = _reverb_noise(wav)
    return wav


gutils.AudioDataset.__init__, gutils.AudioDataset._load_audio = _patched_init, _patched_load

# ---- GPU efficiency ----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Not fused AdamW: Lightning refuses gradient clipping with fused optimizers, and we keep clipping (1.0).
# PyTorch's default multi-tensor (foreach) AdamW is used on CUDA, close in speed.
BATCH_SECONDS = float(os.environ.get("BATCH_SECONDS", "1200"))  # padded audio seconds per training batch


class DurationBatchSampler:
    """Shuffle, then sort inside pools of 64 batches by duration, cut batches by padded cost
    (max_dur * n <= BATCH_SECONDS), and shuffle the batch order. Little padding, still random."""

    def __init__(self, durations, max_seconds, seed=42):
        self.d, self.max_s, self.seed, self.epoch = list(durations), max_seconds, seed, 0
        self._n = sum(1 for _ in self._batches(0))
        self.seconds_yielded = 0.0

    def _batches(self, epoch):
        import numpy as np
        rng = np.random.default_rng(self.seed + epoch)
        order = rng.permutation(len(self.d))
        avg = sum(self.d) / max(len(self.d), 1)
        pool = max(1000, int(64 * self.max_s / max(avg, 0.1)))
        out = []
        for i in range(0, len(order), pool):
            chunk = sorted(order[i:i + pool], key=lambda j: self.d[j])
            b, mx = [], 0.0
            for j in chunk:
                m = max(mx, self.d[j])
                if b and m * (len(b) + 1) > self.max_s:
                    out.append(b); b, m = [], self.d[j]
                b.append(int(j)); mx = m
            if b:
                out.append(b)
        rng.shuffle(out)
        return out

    def __iter__(self):
        for b in self._batches(self.epoch):
            self.seconds_yielded += sum(self.d[j] for j in b)
            yield b
        self.epoch += 1

    def __len__(self):
        return self._n


_DL = torch.utils.data.DataLoader
TRAIN_SAMPLER = []


class EfficientDataLoader(_DL):
    def __init__(self, dataset, *args, **kw):
        if getattr(dataset, "_train", False) and BATCH_SECONDS > 0:
            for k in ("batch_size", "shuffle", "drop_last", "sampler"):
                kw.pop(k, None)
            kw["batch_sampler"] = DurationBatchSampler([x.duration for x in dataset.samples], BATCH_SECONDS)
            TRAIN_SAMPLER[:] = [kw["batch_sampler"]]
            print(f"[wrapper] duration-bucketed batches: {BATCH_SECONDS:.0f} padded s, "
                  f"{len(kw['batch_sampler'])} batches/epoch", flush=True)
        if kw.get("num_workers", 0) > 0:
            kw.setdefault("persistent_workers", True)
            kw.setdefault("prefetch_factor", 4)
        super().__init__(dataset, *args, **kw)


torch.utils.data.DataLoader = EfficientDataLoader

import time

import pytorch_lightning as pl


class Throughput(pl.Callback):
    """Every N steps: step time, audio-s/s, dataloader-wait share, peak memory. One sync per report."""

    def __init__(self, every=int(os.environ.get("TP_EVERY", "50"))):
        self.every, self.t0, self.wait, self.last_end, self.sec0 = every, None, 0.0, None, 0.0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        if self.last_end is not None:
            self.wait += now - self.last_end
        if self.t0 is None:
            self.t0 = now
            self.sec0 = TRAIN_SAMPLER[0].seconds_yielded if TRAIN_SAMPLER else 0.0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.last_end = time.perf_counter()
        step = trainer.global_step
        if step and step % self.every == 0 and self.t0 is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - self.t0
            sec = (TRAIN_SAMPLER[0].seconds_yielded if TRAIN_SAMPLER else 0.0) - self.sec0
            print(f"[tp] {time.strftime('%H:%M:%S', time.gmtime())}Z step {step} | {dt / self.every:.3f} s/step | {sec / dt:.0f} audio-s/s "
                  f"(approx) | data-wait {100 * self.wait / dt:.0f}% | peak mem "
                  f"{(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0) / 2**30:.1f} GiB", flush=True)
            self.t0, self.wait = time.perf_counter(), 0.0
            self.sec0 = TRAIN_SAMPLER[0].seconds_yielded if TRAIN_SAMPLER else 0.0


class LoadTeacher(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        if KD_WEIGHT > 0 and KD_TEACHER and KD["teacher"] is None:
            t = gigaam.load_model(KD_TEACHER, fp16_encoder=False, device=pl_module.device)
            t.eval()
            for p in t.parameters():
                p.requires_grad = False
            t.encoder._force_full = True
            KD["teacher"] = t
            print(f"[wrapper] KD teacher loaded on {pl_module.device}, full context forced", flush=True)


class EMA(pl.Callback):
    """Exponential moving average of the trainable weights (fp32 shadow, updated after every optimizer step).
    At the end of training it writes a checkpoint gigaam.load_model() can read (raw weights replaced by EMA)."""

    def __init__(self, decay, out):
        self.decay, self.out = decay, out

    def on_train_start(self, trainer, pl_module):
        named = dict(pl_module.named_parameters())
        self.names = [n for n, p in named.items() if p.requires_grad and not n.startswith("preprocessor.")]
        self.params = [named[n] for n in self.names]
        self.shadow = [p.detach().float().clone() for p in self.params]
        print(f"[wrapper] EMA decay {self.decay} over {len(self.names)} tensors -> {self.out}", flush=True)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch._foreach_mul_(self.shadow, self.decay)
        torch._foreach_add_(self.shadow, [p.detach().float() for p in self.params], alpha=1 - self.decay)

    def on_train_end(self, trainer, pl_module):
        sd = {k: v.detach().cpu() for k, v in pl_module.state_dict().items()}
        for n, sh in zip(self.names, self.shadow):
            sd[n] = sh.to(sd[n].dtype).cpu()
        tmp = self.out + ".tmp"
        torch.save({"state_dict": sd, "hyper_parameters": dict(pl_module.hparams)}, tmp)
        os.replace(tmp, self.out)
        print(f"[wrapper] EMA checkpoint written: {self.out}", flush=True)


class TrainMode(pl.Callback):
    """gigaam.load_model() returns the model in eval mode and Lightning >= 2.2 preserves submodule modes, so without
    this the encoder/head train in eval mode: activation checkpointing silently off, and the streaming patch's dynamic
    chunk sampling (which keys on encoder.training) never runs. GigaAMFineTuner.train() keeps the frozen preprocessor
    in eval. Lightning restores these modes after each validation."""

    def on_train_start(self, trainer, pl_module):
        before = sum(not m.training for m in pl_module.modules())
        if os.environ.get("FORCE_TRAIN_MODE", "1") == "1":
            pl_module.train()
        after = sum(not m.training for m in pl_module.modules())
        print(f"[wrapper] modes at train start: root={pl_module.training} encoder={pl_module.encoder.training} "
              f"head={pl_module.head.training} preprocessor={pl_module.preprocessor.training} | modules in eval: "
              f"{before} -> {after}", flush=True)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if os.environ.get("DEBUG_MODES") == "1" and batch_idx < 3:
            print(f"[debug] step {batch_idx}: root={pl_module.training} encoder={pl_module.encoder.training} "
                  f"spec_augment={'on' if (pl_module.training and pl_module._spec_augment) else 'OFF'} "
                  f"activation_checkpointing={pl_module.encoder.activation_checkpointing and pl_module.encoder.training}",
                  flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if os.environ.get("DEBUG_MODES") == "1" and batch_idx < 3:
            print(f"[debug] step {batch_idx}: encoder chunk used = {getattr(pl_module.encoder, '_chunk', 'n/a')}", flush=True)


EMA_DECAY, EMA_OUT = float(os.environ.get("EMA_DECAY", "0")), os.environ.get("EMA_OUT", "")
_TInit = pl.Trainer.__init__


def _trainer_init(self, *args, **kw):
    extra = [TrainMode(), LoadTeacher(), Throughput()] + ([EMA(EMA_DECAY, EMA_OUT)] if EMA_DECAY > 0 and EMA_OUT else [])
    kw["callbacks"] = list(kw.get("callbacks") or []) + extra
    _TInit(self, *args, **kw)


pl.Trainer.__init__ = _trainer_init
print(f"[wrapper] speed perturbation 0.9/1.0/1.1 on train manifests: {SPEED}", flush=True)

if os.environ.get("CHUNK_TRAIN") == "1":  # option C: chunk-aware streaming fine-tune
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "service"))
    import streaming_patch
    streaming_patch.enable()

sys.path.insert(0, f"{GIGAAM_DIR}/train_utils")
os.chdir(f"{GIGAAM_DIR}/train_utils")

# ---- knowledge distillation (stage C v2): KD_TEACHER = frozen full-context model, KD_WEIGHT = weight of the
# frame-level KL(teacher || student) on CTC posteriors, added to the CTC loss. The teacher sees the clean features at
# full context (its encoder has _force_full, so the streaming patch leaves it alone); the student sees SpecAugmented
# features under its chunk mask. The teacher is kept outside the module tree: never saved, optimized or mode-switched.
KD_WEIGHT, KD_TEACHER = float(os.environ.get("KD_WEIGHT", "0")), os.environ.get("KD_TEACHER", "")
# KD_GATE=1 (C v2b): distil only where the student can match a full-context teacher's timing: full-context batches and
# chunked batches with lookahead (R > 0). Chunked R=0 batches train on CTC alone, because pulling their CTC spikes
# toward a teacher that sees the future is what the step-750 val of C v2 suggested hurts R=0.
KD_GATE = os.environ.get("KD_GATE", "0") == "1"
KD = {"teacher": None}
if KD_WEIGHT > 0 and KD_TEACHER:
    import module as gmod

    def kd_training_step(self, batch, batch_idx):
        wavs, wav_lens, tokens, tok_lens = batch
        amp_dev = next(self.parameters()).device.type
        with torch.amp.autocast(amp_dev, enabled=False):
            feats, feat_lens = self.preprocessor(wavs.float(), wav_lens)
        sfeats = feats.clone()
        if self.training and self._spec_augment:
            for aug in self._freq_aug:
                sfeats = aug(sfeats)
            for aug in self._time_aug:
                sfeats = aug(sfeats)
        encoded, enc_lens = self.encoder(sfeats, feat_lens)
        log_probs = self.head(encoded)
        ctc = self._ctc_loss(log_probs, tokens, enc_lens, tok_lens)
        assert not KD["teacher"].training and not any(m.training for m in KD["teacher"].modules()), "KD teacher not in eval"
        with torch.no_grad():
            t_enc, _ = KD["teacher"].encoder(feats, feat_lens)
            t_lp = KD["teacher"].head(t_enc).float()
        s_lp = log_probs.float()
        n = min(t_lp.shape[1], s_lp.shape[1])
        valid = (torch.arange(n, device=s_lp.device)[None, :] < enc_lens[:, None]).float()
        kl = (t_lp[:, :n].exp() * (t_lp[:, :n] - s_lp[:, :n])).sum(-1)
        kl = (kl * valid).sum() / valid.sum().clamp(min=1)
        c_, r_ = getattr(self.encoder, "_chunk", (0, 0))
        use_kd = (not KD_GATE) or c_ == 0 or r_ > 0
        loss = ctc + KD_WEIGHT * kl if use_kd else ctc
        self.log("train/loss", loss)
        self.log("train/ctc", ctc.detach())
        self.log("train/kd_kl", kl.detach())
        if batch_idx % 100 == 0:  # emission-delay proxy: centroid of non-blank frames, student minus teacher
            with torch.no_grad():
                idx = torch.arange(n, device=s_lp.device)[None, :].float()
                def centroid(lp):
                    nb = (lp[:, :n].argmax(-1) != self._blank_id).float() * valid
                    return (nb * idx).sum(1) / nb.sum(1).clamp(min=1)
                delay = (centroid(s_lp) - centroid(t_lp)).mean().item()
            print(f"[kd] step {batch_idx}: ctc {ctc.item():.3f} kl {kl.item():.3f}{'' if use_kd else ' (not applied)'} spike delay {delay * 40:+.0f} ms "
                  f"chunk {getattr(self.encoder, '_chunk', None)} teacher {getattr(KD['teacher'].encoder, '_chunk', None)}",
                  flush=True)
        return loss

    gmod.GigaAMFineTuner.training_step = kd_training_step
    print(f"[wrapper] KD on: teacher {KD_TEACHER}, weight {KD_WEIGHT}, gate {'full-context or lookahead batches only' if KD_GATE else 'off (all batches)'}", flush=True)
sys.argv = ["train.py"] + sys.argv[1:]
runpy.run_path("train.py", run_name="__main__")
