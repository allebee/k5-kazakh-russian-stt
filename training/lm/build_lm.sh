#!/usr/bin/env bash
# KenLM 4-grams for the final pass, as released (lm/{kk,ru,kkru}.bin + lm/{kk,ru,kkru}.unigrams.txt):
#   kk   = CC-100 kk + Kazakh training transcripts (KSC2, KSD, FLEURS train)
#   ru   = Russian training transcripts (Golos, SOVA, RuLS, FLEURS train) + the first ~8 GiB of CC-100 ru
#          (cleaned text capped at 1.5e9 tokens)
#   kkru = kk + ru
# Text gets the eval normalization (lowercase, ё->е, punctuation dropped), restricted to GigaAM's vocabulary; lines
# with digits are dropped, and so is every line sharing a word 8-gram with FLEURS dev/test or FLORES-200 dev/devtest
# (kaz_Cyrl, rus_Cyrl). 4-gram, pruning 0 0 1 1, binary trie with 8-bit quantization. Unigram lists = words seen >= 3
# times (pyctcdecode needs them; binary KenLM files don't list words).
#
# Needs: lmplz and build_binary from KenLM (https://github.com/kpu/kenlm), on PATH or in $KENLM_BIN; xz, curl;
# manifests/*.tsv from training/build_manifests.py; models/gigaam-ml-ctc/config.json (vocabulary). Run in the
# workspace ($K5_STT_ROOT). Dev-tuned alpha/beta: eval/lm_eval.py (writes eval/lm_<tag>.json = lm_weights.json on HF).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "${K5_STT_ROOT:-.}"; export K5_STT_ROOT=$PWD
PY=${PY:-python}; B=${KENLM_BIN:+$KENLM_BIN/}
T=data/text; mkdir -p $T/flores lm

# FLORES-200 dev/devtest (decontamination only)
if ! find $T/flores -name "*Cyrl*" -type f 2>/dev/null | grep -q devtest; then
  [ -s $T/flores/flores200_dataset.tar.gz ] || curl -sSfL -o $T/flores/flores200_dataset.tar.gz https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
  tar -xzf $T/flores/flores200_dataset.tar.gz -C $T/flores --wildcards '*kaz_Cyrl*' '*rus_Cyrl*'
fi
find $T/flores -name "*Cyrl*" -type f | grep -q devtest || { echo "FLORES-200 missing: refusing to build an undecontaminated LM"; exit 1; }

# CC-100 (https://data.statmt.org/cc-100/): all of kk, the first 8 GiB of the ru archive (a truncated .xz is fine)
[ -s $T/cc100_kk.txt.xz ] || curl -sSfL -o $T/cc100_kk.txt.xz https://data.statmt.org/cc-100/kk.txt.xz
[ -s $T/cc100_ru_part.txt.xz ] || curl -sSfL -r 0-8589934591 -o $T/cc100_ru_part.txt.xz https://data.statmt.org/cc-100/ru.txt.xz

# text: lm_text.py writes lm_kk.txt, lm_ru.txt (transcripts only) and lm_kkru.txt; the CC-100 ru part is added after
[ -s $T/lm_kk.txt ] || $PY "$HERE/lm_text.py"
[ -s $T/lm_ru_cc100.txt ] || $PY "$HERE/lm_text_ru.py"
[ -s $T/lm_ru_transcripts.txt ] || mv $T/lm_ru.txt $T/lm_ru_transcripts.txt
cat $T/lm_ru_transcripts.txt $T/lm_ru_cc100.txt > $T/lm_ru.txt
cat $T/lm_kk.txt $T/lm_ru.txt > $T/lm_kkru.txt

for n in kk ru kkru; do
  if [ ! -s lm/$n.bin ]; then
    ${B}lmplz -o 4 --prune 0 0 1 1 -S ${LMPLZ_MEM:-30%} -T "$PWD/lm" --discount_fallback < $T/lm_$n.txt > lm/$n.arpa
    ${B}build_binary -q 8 -b 8 trie lm/$n.arpa lm/$n.bin && rm lm/$n.arpa
  fi
  [ -s lm/$n.unigrams.txt ] || LC_ALL=C tr -s ' ' '\n' < $T/lm_$n.txt | LC_ALL=C sort -S ${SORT_MEM:-16G} -T lm | uniq -c \
    | awk '$1>=3{print $2}' > lm/$n.unigrams.txt
  echo "$n: $(du -h lm/$n.bin | cut -f1), $(wc -l < lm/$n.unigrams.txt) unigrams"
done
echo LM_BUILD_DONE
