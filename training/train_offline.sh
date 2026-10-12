#!/usr/bin/env bash
# Offline fine-tune of GigaAM-Multilingual large_ctc: the recipe behind the released offline (final-pass) model.
#   1. ONE pass (epoch mode, validation every 1/4) over manifests/train_full.tsv, LR 3e-5, bf16-mixed,
#      ~3600 padded audio-seconds per batch, with an EMA (decay 0.9995) kept alongside the raw weights
#   2. WiSE-FT candidates alpha 0.75 / 0.50 / 0.25, built from the EMA weights
#   3. greedy evals of every candidate on the DEV sets, selection by mean no-digit dev WER (select_alpha.py);
#      test sets are never used for selection. In our run the raw end-of-pass checkpoint (final_a1.00.ckpt) won.
#   4. greedy evals of all candidates on the test sets that exist under eval/manifests/
#
# Run with a Python that has the gigaam package installed from a GigaAM clone with its training extras
# (pip install -e "GigaAM[train]"). Paths: K5_STT_ROOT = workspace (default: current dir), GIGAAM_DIR = the clone.
# Prerequisites in the workspace: manifests/*.tsv (training/build_manifests.py, training/make_augment.py) and the dev
# manifests eval/manifests/dev_*.jsonl (eval/prep_eval.py, eval/eval_extra.py, eval/make_dev_sets.py).
# Idempotent: finished stages are skipped on re-run.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE")
cd "${K5_STT_ROOT:-.}"; export K5_STT_ROOT=$PWD
PY=${PY:-python}
MODEL=${MODEL:-multilingual_large_ctc}
export MEM_FRACTION=${MEM_FRACTION:-0.75} BATCH_SECONDS=${BATCH_SECONDS:-3600}
LR=${LR:-3e-5}
RUN=$PWD/runs/full_$MODEL; FINAL=$RUN/final_a1.00.ckpt; EMA=$RUN/ema_final.ckpt; TAG=ft-${MODEL/multilingual_/}
export EMA_DECAY=${EMA_DECAY:-0.9995} EMA_OUT=$EMA
log(){ echo "[$(date -u '+%F %T')] $*"; }
complete(){ [ -s "$1" ] && [ "$(wc -l < "$1")" -eq "$(wc -l < "$2")" ]; }
pairs(){ local tag=$1; shift; for s in "$@"; do [ -f eval/manifests/$s.jsonl ] && echo -n " eval/manifests/$s.jsonl=eval/hyps/$s.$tag.greedy.jsonl"; done; }
DEV="dev_fleurs_kk dev_fleurs_ru dev_fleurs_kk_tel dev_fleurs_ru_tel dev_ksc2_spont"
TEST="fleurs_kk fleurs_kk_tel fleurs_ru fleurs_ru_tel cs_tim2190 $(cd eval/manifests && ls cs_ksc2.jsonl ksc2_*.jsonl golos_*.jsonl 2>/dev/null | sed 's/.jsonl//' | tr '\n' ' ')"
mkdir -p "$RUN" eval/hyps
[ -s manifests/train_full.tsv ] || $PY "$HERE/prep_full_manifest.py"
for f in tel_kk tel_ru cs_concat val; do [ -s manifests/$f.tsv ] || { echo "manifests/$f.tsv missing"; exit 1; }; done

# ---- 1. one pass of fine-tuning (+ EMA) ----
if [ -s "$FINAL" ] && { [ -s "$EMA" ] || [ "$EMA_DECAY" = 0 ]; }; then log "skip training: $FINAL exists"; else
  log "train: $MODEL, $BATCH_SECONDS padded s/batch, LR $LR, EMA $EMA_DECAY"
  $PY "$HERE/train_wrapper.py" --model_name $MODEL \
    --train_manifest $PWD/manifests/train_full.tsv --val_manifest $PWD/manifests/val.tsv \
    --output_dir $RUN --exp_name ft --raw_text --precision bf16-mixed --activation_checkpointing \
    --batch_size 64 --eval_batch_size 32 --num_workers 16 --lr $LR --disable_tqdm \
    --max_epochs 1 --val_check_interval 0.25 --val_first_batches 20 --log_every_n_steps 100 --save_top_k 4
  CK=$(find "$RUN" -name "gigaam-*.ckpt" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
  [ -n "$CK" ] && [ -s "$CK" ] || { echo "the trainer wrote no checkpoint under $RUN"; exit 1; }
  cp "$CK" "$FINAL.tmp" && mv "$FINAL.tmp" "$FINAL"
  log "final (end-of-pass) checkpoint: $FINAL (from $CK)"
fi

# ---- 2. WiSE-FT from the EMA weights (theta = a*theta_EMA + (1-a)*theta_0); from the raw weights if EMA is off ----
SRC=$FINAL; [ -s "$EMA" ] && SRC=$EMA
for a in 0.75 0.50 0.25; do [ -s $RUN/wise_a$a.ckpt ] || $PY "$HERE/wise_ft.py" "$SRC" $MODEL $RUN $a; done
declare -A CKPT=([$TAG-raw]=$FINAL [$TAG-a0.75]=$RUN/wise_a0.75.ckpt [$TAG-a0.50]=$RUN/wise_a0.50.ckpt [$TAG-a0.25]=$RUN/wise_a0.25.ckpt)
[ -s "$EMA" ] && CKPT[$TAG-ema]=$EMA

# ---- 3. dev evals + dev-only selection ----
DEVICE=${DEVICE:-cuda}
for t in "${!CKPT[@]}"; do $PY "$REPO/eval/eval_many.py" "${CKPT[$t]}" $DEVICE $(pairs $t $DEV); done
BEST=$($PY "$HERE/select_alpha.py" $RUN/selected.json "${!CKPT[@]}" | tail -1)
log "selected on dev: $BEST -> ${CKPT[$BEST]}"

# ---- 4. test evals of all candidates (reporting only) ----
for t in "${!CKPT[@]}"; do $PY "$REPO/eval/eval_many.py" "${CKPT[$t]}" $DEVICE $(pairs $t $TEST); done
log "done: score with  python $REPO/eval/score.py eval/manifests/<set>.jsonl eval/hyps/<set>.*.jsonl"
