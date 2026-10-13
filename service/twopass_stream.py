"""Streaming dialogue interface on the two-pass engine (for voice dialogue: speech in, LLM + TTS reply out).

    from twopass import TwoPass, release_paths
    from twopass_stream import StreamSession
    tp = TwoPass(paths=release_paths("k5stt"), device="cuda", lm="beam")   # models, loaded once per process (CPU works too)
    s = StreamSession(tp, eou_ms=400, pause_ms=200, lang="beam", hotwords=["Kaspi=каспи", "Halyk=халық"])
    events = s.feed(pcm_f32)                         # 16 kHz mono float32, any length
    events = s.poll()                                # right after an `eou` event: get the owed final (~20-40 ms on GPU)
    events = s.flush()                               # finalize now (push-to-talk)
    s.reset()                                        # new turn / barge-in (the audio clock keeps running)

Events (dicts); `t` = seconds of audio since the session was created; `proc_ms` = compute spent on the event:
  speech_start {t, t_detect}       VAD onset (start of the first speech window), detected at t_detect
  partial      {text, tentative, t} every chunk (320 ms default) while in a turn (engine fed in >= 160 ms blocks); `text` = committed words (never
                                   rewritten), `tentative` = the still-growing last word
  pause        {text, itn, partial, t, t_speech_end, silence_ms}   first ~200 ms of silence: `text` is an offline-quality
                                   decode of the turn so far (final model + KenLM + hotwords), for a speculative LLM call
  resume       {t}                 speech again after a pause, before the end of the turn (VAD only: breath/noise can do it)
  blip         {t, t_end, windows, in_turn}   VAD speech shorter than speech_start_ms (idle) / resume_ms (paused): no
                                   turn start / resume, the silence timer keeps running (informational)
  words        {t, mark, t_mark, since_mark_ms, new_words, kind}   first real word (>= 2 letters, not a hesitation) in
                                   the partials after a speech_start or resume: the "speaker really said something" signal
  eou          {t, t_speech_end, resumed, early}   end of turn detected; emitted BEFORE the final decode (feed returns
                                   right after it); the final follows from poll() or the next feed()
  final        {text, itn, t, t_speech_end, early, same_as_pause, forced}   end of turn (EOU silence, semantic early
                                   endpoint, flush, or max_turn_s), offline final pass + KenLM + hotwords + ITN

VAD: Silero vad.onnx (the model repo's vad/) with AGC on 96 ms windows, so pause/EOU/onset resolve to multiples of 96 ms
(pause_ms 200 -> 192 ms, eou_ms 400 -> 384 ms, speech_start_ms 192 -> 2 windows). vad_thr starts a turn and resumes
after a pause; vad_thr_off (default = vad_thr) keeps a running turn in speech (hysteresis). A turn starts after
speech_start_ms and resumes after resume_ms of consecutive windows above vad_thr; a shorter run is reported as `blip`
and does not reset the pause/EOU silence timer (EOU waits until a pending run resolves: at most resume_ms - 96 ms).
cont_ms > 0: within cont_ms after the speech end of a final with real words, one window above cont_thr starts the next
turn (a continuation after a pause longer than eou_ms is a new speech_start, not a resume).
Thread safety: one StreamSession per dialog, used from one thread; any number of sessions may share one TwoPass.
Model compute is serialized by a module lock (no CPU oversubscription, no concurrent GPU kernels from threads).
"""
import os, re, threading, time
from pathlib import Path

import numpy as np
import torch

from twopass import SR

_LOCK = threading.Lock()
_SHARED = {"vad": None, "decoders": {}}
_SHARED_LOCK = threading.Lock()
WIN = 1536                      # VAD window (96 ms at 16 kHz), as in twopass.StreamVAD
W = WIN / SR

# Semantic end-of-turn cue (v1 heuristic, OFF by default): Kazakh question particles and common finite-verb endings /
# closers as the last word of the turn. Measured on dev calls: no gain over a fixed wait (docs/dialog_turn_taking.md).
_KK_PARTICLES = {"ма", "ме", "ба", "бе", "па", "пе", "ше", "ғой", "қой", "екен", "еken"}
_CLOSERS = {"рахмет", "рақмет", "болды", "жарайды", "иә", "жоқ", "керек", "емес", "спасибо", "всё", "все", "да", "нет",
            "хорошо", "пожалуйста", "до свидания", "сау болыңыз"}
_FINITE = re.compile(r"(мын|мін|бын|бін|пын|пін|мыз|міз|быз|біз|пыз|піз|сыңдар|сіңдер|сыздар|сіздер|сың|сің|сыз|сіз|"
                     r"ады|еді|йды|йді|ды|ді|ты|ті|мақпын|мекпін|ған|ген|қан|кен|атын|етін|йтын|йтін|ейік|айық|йық|йік|"
                     r"шы|ші|ңыз|ңіз|ыңыз|іңіз)$")


# Hesitations and backchannels (kk + ru), in the model's output form (lowercase, ё -> е). `classify` gives
#   "filler"      only hesitation sounds (э, мм, ыыы, ну, ...)
#   "backchannel" only acknowledgements / yes-no (иә, жоқ, жақсы, ага, угу, да, нет, ...), possibly with fillers
#   "speech"      anything else; "" for empty text
FILLERS = {"э", "ээ", "эээ", "ә", "әә", "әәә", "а", "аа", "ааа", "м", "мм", "ммм", "хм", "хмм", "ым", "ыы", "ыыы", "ыыыы",
           "е", "ее", "еее", "у", "уу", "ну", "мхм", "эм", "эмм", "ой"}
BACKCHANNELS = {"иә", "ия", "иа", "жоқ", "жақсы", "жарайды", "дұрыс", "рас", "солай", "түсінікті", "рахмет", "рақмет",
                "ага", "угу", "ага ага", "да", "нет", "ок", "окей", "хорошо", "понятно", "ясно", "конечно", "спасибо"}


def classify(text):
    w = text.split()
    if not w:
        return ""
    if all(x in FILLERS for x in w):
        return "filler"
    if all(x in FILLERS or x in BACKCHANNELS for x in w):
        return "backchannel"
    return "speech"


def _real_words(words):
    """Words that count as 'the speaker said something': >= 2 letters and not a pure hesitation."""
    return [x for x in words if len(x) >= 2 and x not in FILLERS]


def looks_final(text):
    w = text.split()
    if not w:
        return False
    last = w[-1]
    return last in _KK_PARTICLES or last in _CLOSERS or " ".join(w[-2:]) in _CLOSERS or bool(_FINITE.search(last))


def _vad_session(tp):
    with _SHARED_LOCK:
        if _SHARED["vad"] is None:
            import onnxruntime as ort
            o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
            _SHARED["vad"] = ort.InferenceSession(str(tp.p["vad"]), o)
        return _SHARED["vad"]


def _decoder(tp, lm, boundary=True, beta=None):
    """Shared pyctcdecode decoders per (LM, boundary scoring, beta), built lazily from tp's paths.
    lm: 'kkru' | 'kk' | 'ru' (KenLM) or 'beam' (beam search without an LM: hotwords still work).
    boundary=False turns off pyctcdecode's sentence-start/-end LM scores; beta overrides the dev-tuned word bonus."""
    if lm in (None, "none"):
        return None
    key = (lm, bool(boundary), beta)
    with _SHARED_LOCK:
        if key not in _SHARED["decoders"]:
            from pyctcdecode import build_ctcdecoder
            if lm == "beam":
                _SHARED["decoders"][key] = build_ctcdecoder(tp.vocab + [""])
            elif boundary and beta is None and tp.dec is not None and getattr(tp, "lm", "kkru") == lm:
                _SHARED["decoders"][key] = tp.dec
            else:
                import json
                a, b = json.load(open(tp.p["lm_weights"])).get(lm) or [0.2, 0.0]
                uni = Path(tp.p["lm_dir"]) / f"{lm}.unigrams.txt"
                dec = build_ctcdecoder(
                    tp.vocab + [""], kenlm_model_path=str(Path(tp.p["lm_dir"]) / f"{lm}.bin"),
                    unigrams=uni.read_text(encoding="utf-8").split() if uni.exists() else None, alpha=a,
                    beta=b if beta is None else beta)
                if not boundary:
                    dec._language_model.score_boundary = False   # LanguageModel attribute (pyctcdecode)
                _SHARED["decoders"][key] = dec
        return _SHARED["decoders"][key]


def parse_hotwords(entries):
    """["Kaspi=каспи|kaspi", "Halyk=халық", "Ерболат"] -> (bias list of spoken forms, display map spoken -> written)."""
    bias, display = [], {}
    for e in entries or []:
        if not e or not e.strip():
            continue
        written, _, spoken = e.partition("=")
        forms = [f for f in (spoken.split("|") if spoken else [written]) if f.strip()]
        for f in forms:
            s = " ".join(f.lower().replace("ё", "е").split())
            bias.append(s)
            if spoken and written.strip():
                display[s] = written.strip()
    return bias, display


class StreamSession:
    def __init__(self, tp, eou_ms=400, pause_ms=200, speech_start_ms=192, lang="auto", hotwords=None,
                 hotword_weight=10.0, pad_ms=500, max_turn_s=20.0, chunk_ms=320, semantic=False, early_eou_ms=288,
                 vad_thr=0.5, beam_width=64, min_feed_ms=160, pause_lm=True, lm_boundary=True, lm_beta=None,
                 agc_max=30.0, vad_thr_off=None, resume_ms=96, cont_ms=0, cont_thr=0.5):
        self.tp = tp
        n = lambda ms: max(1, int(round(ms / 1000 / W)))
        self.n_pause, self.n_eou, self.n_start, self.n_early = n(pause_ms), n(eou_ms), n(speech_start_ms), n(early_eou_ms)
        self.n_pause = min(self.n_pause, self.n_eou)
        self.n_resume = n(resume_ms)
        self.pad, self.max_turn, self.C = pad_ms / 1000, max_turn_s, chunk_ms // 40
        self.semantic, self.thr, self.beam_width = semantic, vad_thr, beam_width
        # VAD input gain cap. 30 (default, validated on offline calls) catches very quiet recordings but also boosts room
        # noise. agc_max=3 cuts noise false starts in eval/dialog/vad_noise_eval.py, but on the dev calls it deletes quiet
        # speech (final WER 12.67 vs 10.54; 15.11 together with vad_thr 0.7), so keep 30 (2026-10-13)
        self.agc_max = float(agc_max)
        # VAD hysteresis: vad_thr starts a turn / resumes after a pause; inside a turn (before the pause) a window counts
        # as speech above vad_thr_off (default = vad_thr, i.e. no hysteresis)
        self.thr_off = self.thr if vad_thr_off is None else min(float(vad_thr_off), self.thr)
        self.min_feed = min_feed_ms / 1000   # partial engine is fed in blocks: per-frame calls cost GPU launch overhead
        self.pause_lm = pause_lm             # False: greedy pause decode (no KenLM beam), slightly faster
        self.lm = {"auto": "kkru", "kk": "kk", "ru": "ru", "none": "none", "beam": "beam"}.get(lang, lang)
        self.dec = _decoder(tp, self.lm, lm_boundary, lm_beta)
        self.set_hotwords(hotwords, hotword_weight)
        self.vad = _vad_session(tp)
        self.clock = 0.0                      # audio seconds since session creation (not reset by reset())
        # continuation window: for cont_ms after the speech end of a final with real words, a new turn starts on ONE window
        # above cont_thr (the old 96 ms / 0.5 rule), so "speaker continues after a long pause" is seen as fast as before while
        # cold starts keep speech_start_ms / vad_thr (a dialog's release timer races this onset)
        self.cont_ms, self.cont_thr, self.cont_until = cont_ms / 1000, float(cont_thr), -1.0
        self._last_wait, self._last_parts = 0.0, {}
        self.reset()

    def set_hotwords(self, entries, weight=10.0):
        self.hotwords, self.display = parse_hotwords(entries)
        self.hotword_weight = float(weight)

    def reset(self):
        """New turn / barge-in: drop the current turn and VAD state; the audio clock keeps running."""
        self.h = self.c = np.zeros((2, 1, 64), np.float32)
        self.rms, self.vbuf = None, np.zeros(0, np.float32)
        self.vad_t = self.clock               # audio time of the next VAD window start
        self.run = self.sil = 0
        self.audio, self.a0 = np.zeros(0, np.float32), self.clock
        self.turn0 = None                     # current turn start (s), None = idle
        self.prev_end = self.clock
        self.eng, self.lps, self.fed = None, [], None
        self.paused, self.pause_res, self.resumed = False, None, False
        self.last_speech_end = None
        self.pending = None                   # (t_speech_end, t_now, early): final owed after an `eou` event
        self.mark = None                      # (kind, t_mark, n_real_words_at_mark, fired): `words` event bookkeeping
        self.words_since_resume = False

    # ---- helpers ------------------------------------------------------------------------------------------
    def _slice(self, t0, t1):
        i0, i1 = int(round((t0 - self.a0) * SR)), int(round((t1 - self.a0) * SR))
        return self.audio[max(0, i0):max(0, i1)]

    def _greedy_words(self):
        lp = torch.cat(self.lps) if self.lps else torch.zeros(0, len(self.tp.vocab) + 1)
        out, prev = [], -1
        for i in lp.argmax(-1).tolist():
            if i != prev and i != len(self.tp.vocab):
                out.append(self.tp.vocab[i])
            prev = i
        return "".join(out).split()

    def _apply_display(self, itn):
        for spoken, written in self.display.items():
            itn = re.sub(rf"(?<!\w){re.escape(spoken)}(?!\w)", written, itn, flags=re.IGNORECASE)
        return itn

    @torch.inference_mode()
    def _decode(self, audio, use_lm=True):
        """Offline final model on `audio` + KenLM beam (+ hotwords) + ITN -> (text, itn, ms)."""
        tp = self.tp
        tw = time.perf_counter()
        os.environ["CHUNK_EVAL"], os.environ["LOOKAHEAD_EVAL"] = "0", "0"
        with _LOCK:
            t0 = time.perf_counter()
            x = torch.from_numpy(np.ascontiguousarray(audio))[None].to(tp.dev)
            if tp.dev.type == "cuda":
                enc, _ = tp.fm.forward(x, torch.tensor([x.shape[1]], device=tp.dev))
                with torch.autocast("cuda", torch.float16):
                    lp = tp.fm.head(enc)[0].float().cpu().numpy()
            else:
                feats, fl = tp.fm.preprocessor(x, torch.tensor([x.shape[1]]))
                with torch.autocast("cpu", torch.bfloat16, enabled=tp.bf16_final):
                    enc, _ = tp.fm.encoder(feats, fl)
                    lp = tp.fm.head(enc)[0].float().numpy()
        t_am = time.perf_counter()
        if self.dec is not None and use_lm:
            kw = dict(hotwords=self.hotwords, hotword_weight=self.hotword_weight) if self.hotwords else {}
            text = self.dec.decode(lp, beam_width=self.beam_width, **kw)
        else:
            out, prev = [], -1
            for i in lp.argmax(-1).tolist():
                if i != prev and i != len(tp.vocab):
                    out.append(tp.vocab[i])
                prev = i
            text = " ".join("".join(out).split())
        t_lm = time.perf_counter()
        itn = tp.itn(text) if tp.itn else text
        self._last_wait = round(1000 * (t0 - tw), 1)
        self._last_parts = {"am_ms": round(1000 * (t_am - t0), 1), "lm_ms": round(1000 * (t_lm - t_am), 1),
                            "itn_ms": round(1000 * (time.perf_counter() - t_lm), 1)}
        return text, self._apply_display(itn), round(1000 * (time.perf_counter() - t0), 1)

    # ---- turn lifecycle -------------------------------------------------------------------------------------
    def _start_turn(self, t_onset, t_detect, ev):
        from cache_engine import CacheEngine
        self.turn0 = max(t_onset - self.pad, self.prev_end, self.a0)
        self.eng, self.lps, self.fed = CacheEngine(self.tp.cm, self.C, cpu_bf16=self.tp.bf16_partials), [], self.turn0
        self.paused, self.pause_res, self.resumed = False, None, False
        ev.append({"type": "speech_start", "t": round(t_onset, 3), "t_detect": round(t_detect, 3)})
        self.mark, self.words_since_resume = ("speech_start", t_onset, 0, False), False

    def _partials(self, upto, ev):
        x = self._slice(self.fed, upto)
        self.fed = upto
        if not len(x):
            return
        tw = time.perf_counter()
        with _LOCK, torch.inference_mode():
            t0 = time.perf_counter()
            new = self.eng.feed(x)
            if new:
                new = [n.float().cpu() for n in new]
        if not new:
            return
        self.lps += new
        words = self._greedy_words()
        ev.append({"type": "partial", "text": " ".join(words[:-1]), "tentative": words[-1] if words else "",
                   "t": round(upto, 3), "proc_ms": round(1000 * (time.perf_counter() - t0), 1),
                   "wait_ms": round(1000 * (t0 - tw), 1), "kind": classify(" ".join(words)),
                   "filler_only": classify(" ".join(words)) in ("filler", "backchannel")})
        if self.mark and not self.mark[3]:
            real = _real_words(words)
            if len(real) > self.mark[2]:
                kind_new = classify(" ".join(real[self.mark[2]:]))
                ev.append({"type": "words", "t": round(upto, 3), "mark": self.mark[0], "t_mark": round(self.mark[1], 3),
                           "since_mark_ms": round(1000 * (upto - self.mark[1])), "new_words": real[self.mark[2]:],
                           "kind": kind_new})
                self.mark = (self.mark[0], self.mark[1], self.mark[2], True)
                if self.mark[0] == "resume":
                    self.words_since_resume = True

    def _final(self, t_speech_end, t_now, ev, early=False, forced=False):
        s1 = min(t_speech_end + self.pad, t_now)
        text, itn, ms = self._decode(self._slice(self.turn0, s1))
        same = bool(self.pause_res) and not self.resumed and text == self.pause_res[0]
        if self.cont_ms and _real_words(text.split()):
            self.cont_until = t_speech_end + self.cont_ms
        ev.append({"type": "final", "text": text, "itn": itn, "t": round(t_now, 3), "t_speech_end": round(t_speech_end, 3),
                   "kind": classify(text), "filler_only": classify(text) in ("filler", "backchannel"),
                   "early": early, "forced": forced, "same_as_pause": same, "proc_ms": ms, "wait_ms": self._last_wait, **self._last_parts})
        self.prev_end, self.turn0, self.eng = s1, None, None
        self.paused, self.pause_res, self.resumed = False, None, False
        self.mark, self.words_since_resume = None, False
        keep_from = max(self.a0, s1 - self.pad)
        self.audio = self._slice(keep_from, self.clock); self.a0 = keep_from

    # ---- public API -------------------------------------------------------------------------------------------
    def feed(self, pcm):
        """Add audio; returns events. An `eou` event ends the call early: the owed final comes from poll() (call it
        right away) or from the next feed()."""
        pcm = np.asarray(pcm, np.float32).reshape(-1)
        self.audio = np.concatenate([self.audio, pcm]); self.clock += len(pcm) / SR
        self.vbuf = np.concatenate([self.vbuf, pcm])
        return self._process()

    def poll(self):
        """Compute the final owed after an `eou` event (and process any audio still queued)."""
        return self._process()

    def _process(self):
        ev = []
        if self.pending:
            t_end, t_now, early = self.pending; self.pending = None
            self._final(t_end, t_now, ev, early=early)
        while len(self.vbuf) >= WIN:
            win, self.vbuf = self.vbuf[:WIN], self.vbuf[WIN:]
            t = self.vad_t; self.vad_t += W
            r = float(np.sqrt(np.mean(win ** 2)) + 1e-6)
            self.rms = r if self.rms is None else (0.9 * self.rms + 0.1 * r if r < self.rms else 0.5 * self.rms + 0.5 * r)
            g = np.clip(win * min(self.agc_max, 0.1 / self.rms), -1, 1).astype(np.float32) if self.agc_max > 1 else win
            p, self.h, self.c = self.vad.run(None, {"input": g[None], "sr": np.array(SR, np.int64), "h": self.h, "c": self.c})
            in_speech = self.turn0 is not None and not self.paused
            cont = self.turn0 is None and t < self.cont_until
            if float(np.ravel(p)[0]) > (self.thr_off if in_speech else min(self.thr, self.cont_thr) if cont else self.thr):
                self.run += 1
                if self.turn0 is None:
                    if self.run >= (1 if cont else self.n_start):
                        self._start_turn(t - (self.run - 1) * W, t + W, ev)
                elif self.paused and self.run >= self.n_resume:
                    self.paused, self.resumed = False, True
                    ev.append({"type": "resume", "t": round(t + W, 3)})
                    self.mark = ("resume", t - (self.run - 1) * W, len(_real_words(self._greedy_words())), False)
                    self.words_since_resume = False
                if self.turn0 is None or not self.paused:
                    self.sil = 0
                    self.last_speech_end = t + W
                    continue
                # paused, resume not confirmed yet: the window counts as silence (a blip does not reset the timer)
            else:
                if 0 < self.run < (self.n_resume if self.paused else self.n_start) and (self.paused or self.turn0 is None):
                    ev.append({"type": "blip", "t": round(t - self.run * W, 3), "t_end": round(t, 3), "windows": self.run,
                               "in_turn": self.turn0 is not None})
                self.run = 0
                if self.turn0 is None:
                    continue
            self.sil += 1
            t_end = t - (self.sil - 1) * W            # end of the last speech window (unpadded VAD speech end)
            t_now = t + W
            if self.sil == self.n_pause and not self.paused:
                self._partials(t_now, ev)
                text, itn, ms = self._decode(self._slice(self.turn0, t_now), use_lm=self.pause_lm)
                self.paused, self.pause_res = True, (text, itn)
                words = self._greedy_words()
                ev.append({"type": "pause", "text": text, "itn": itn, "partial": " ".join(words), "t": round(t_now, 3),
                           "kind": classify(text), "filler_only": classify(text) in ("filler", "backchannel"),
                           "t_speech_end": round(t_end, 3), "silence_ms": round(1000 * self.sil * W), "proc_ms": ms,
                           "wait_ms": self._last_wait, **self._last_parts})
            early = (self.semantic and self.sil == self.n_early and self.n_early < self.n_eou and self.pause_res
                     and looks_final(self.pause_res[0]))
            if (early or self.sil >= self.n_eou) and not self.run:   # not inside a pending resume run: wait for it
                ev.append({"type": "eou", "t": round(t_now, 3), "t_speech_end": round(t_end, 3),
                           "resumed": self.resumed, "words_since_resume": self.words_since_resume,
                           "early": bool(early)})
                self.pending = (t_end, t_now, bool(early))
                return ev                              # final follows from poll() / the next feed()
        if self.turn0 is not None:
            if not self.paused and self.clock - self.fed >= self.min_feed:
                self._partials(self.clock, ev)
            if self.clock - self.turn0 >= self.max_turn:
                self._final(self.clock, self.clock, ev, forced=True)
                self._start_turn(self.clock, self.clock, ev)
        elif self.clock - self.a0 > 2 * self.pad + 1:     # idle: keep only the pre-roll
            keep_from = self.clock - self.pad
            self.audio = self._slice(keep_from, self.clock); self.a0 = keep_from
        return ev

    def flush(self):
        """Finalize the current turn now (push-to-talk release)."""
        ev = []
        if self.pending:
            t_end, t_now, early = self.pending; self.pending = None
            self._final(t_end, t_now, ev, early=early)
        if self.turn0 is not None:
            t_end = self.last_speech_end or self.clock
            self._final(min(t_end, self.clock), self.clock, ev)
        return ev
