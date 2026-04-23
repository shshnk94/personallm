#!/usr/bin/env bash
# Extract message-level 1grams and dd_depAnx lexicon features from personallm.responses.
# Run from /data/personaLLM/dlatk so dlatkInterface.py resolves locally.
set -euo pipefail

cd "$(dirname "$0")/../dlatk"

PY=/home/ssubrahmanya/.conda/envs/dlatk/bin/python

# Message-level 1grams -> feat$1gram$responses$message_id$16to16
$PY dlatkInterface.py \
    -d personallm \
    -t responses \
    -c message_id \
    --add_ngrams -n 1

# Weighted lexicon features from dlatk_lexica.dd_depAnx -> feat$cat_dd_depAnx_w$responses$message_id$1gra
$PY dlatkInterface.py \
    -d personallm \
    -t responses \
    -c message_id \
    --add_lex_table -l dd_depAnx --weighted_lexicon \
    --lexicondb dlatk_lexica

# LIWC2015 lexicon features from dlatk_lexica.LIWC2015 -> feat$cat_LIWC2015$responses$message_id$1gra
$PY dlatkInterface.py \
    -d personallm \
    -t responses \
    -c message_id \
    --add_lex_table -l LIWC2015 \
    --lexicondb dlatk_lexica
