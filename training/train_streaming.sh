#!/usr/bin/env bash
# Stage C: chunk-aware streaming fine-tune (the released partial model, streaming/ on Hugging Face = "C v1").
#   init: the dev-selected offline checkpoint (STAGE_INIT, default runs/full_<MODEL>/final_a1.00.ckpt)
#   1,500 steps (~0.6 of a pass over train_full.tsv), LR 5e-5, no EMA; streaming_patch with dynamic chunks
#   160/320/640/1280 ms per forward pass (full context with p = 0.15), 4 s of left context, NO lookahead (R = 0);
#   validation under the 640 ms chunk mask.
#   Then chunk-masked greedy evals at 320 / 640 / 1280 ms and full context (= what the cache-aware engine computes).
# Note: a later variant (KD from the offline model, sampled lookahead, EMA, more Russian; env KD_TEACHER/KD_WEIGHT/
# TRAIN_LOOKAHEADS in train_wrapper.py and service/streaming_patch.py) was no better at R = 0, and its lookahead
# numbers were invalid (see README, "Limitations"). The released partial model is this R = 0 stage.
# Same environment and paths as train_offline.sh.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE")
cd "${K5_STT_ROOT:-.}"; export K5_STT_ROOT=$PWD
PY=${PY:-python}
GIGAAM_DIR=$(cd "${GIGAAM_DIR:-$PWD/GigaAM}" && pwd); export GIGAAM_DIR
MODEL=${MODEL:-multilingual_large_ctc}; INIT=${STAGE_INIT:-$PWD/runs/full_$MODEL/final_a1.00.ckpt}
export MEM_FRACTION=${MEM_FRACTION:-0.75} BATCH_SECONDS=${BATCH_SECONDS:-3600}
C_STEPS=${C_STEPS:-1500}; LR_C=${LR_C:-5e-5}
export EMA_DECAY=0; unset EMA_OUT   # stage C runs without EMA
RUNC=$PWD/runs/stream_$MODEL; FINALC=$RUNC/final_stream.ckpt; TAG=stream-${MODEL/multilingual_/}
DEVICE=${DEVICE:-cuda}
log(){ echo "[$(date -u '+%F %T')] [C] $*"; }
[ -s "$INIT" ] || { echo "init checkpoint missing: $INIT"; exit 1; }
mkdir -p "$RUNC" eval/hyps

if [ -s "$FINALC" ]; then log "skip training: $FINALC exists"; else
  # GigaAM's trainer names checkpoints after --model_name, so give the init checkpoint a plain relative name
  ln -sf "$INIT" "$GIGAAM_DIR/train_utils/stt_init_c.ckpt"
  log "train: init $INIT, $C_STEPS steps x $BATCH_SECONDS padded s, LR $LR_C, dynamic chunks; validation at 640 ms chunks"
  CHUNK_TRAIN=1 CHUNK_EVAL=16 $PY "$HERE/train_wrapper.py" --model_name stt_init_c.ckpt \
    --train_manifest $PWD/manifests/train_full.tsv --val_manifest $PWD/manifests/val.tsv \
    --output_dir $RUNC --exp_name ftc --raw_text --precision bf16-mixed --activation_checkpointing \
    --batch_size 64 --eval_batch_size 32 --num_workers 16 --lr $LR_C --disable_tqdm \
    --max_steps $C_STEPS --val_check_steps $((C_STEPS / 2)) --val_first_batches 20 --log_every_n_steps 100 --save_top_k 1
  CK=$(find "$RUNC" -name "gigaam-*.ckpt" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
  [ -n "$CK" ] && [ -s "$CK" ] || { echo "no checkpoint written under $RUNC"; exit 1; }
  cp "$CK" "$FINALC.tmp" && mv "$FINALC.tmp" "$FINALC"
  log "final streaming checkpoint: $FINALC"
fi

# chunk-masked evals: chunk 0 (full), 8 / 16 / 32 frames = 320 / 640 / 1280 ms, lookahead 0
SETS="dev_fleurs_kk dev_fleurs_ru dev_fleurs_kk_tel dev_fleurs_ru_tel dev_ksc2_spont fleurs_kk fleurs_ru fleurs_kk_tel fleurs_ru_tel cs_tim2190 cs_ksc2 ksc2_podcasts"
for c in 0 8 16 32; do
  args=""; for s in $SETS; do [ -f eval/manifests/$s.jsonl ] && args="$args eval/manifests/$s.jsonl=eval/hyps/$s.$TAG-c$c.greedy.jsonl"; done
  STREAM_PATCH=1 CHUNK_EVAL=$c LOOKAHEAD_EVAL=0 $PY "$REPO/eval/eval_many.py" "$FINALC" $DEVICE $args
done
log "STAGE_C_DONE"
