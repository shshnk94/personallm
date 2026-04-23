#!/usr/bin/env bash
# Score every personallm.responses row with the pretrained PHQ depression model
# dep_text_PHQ_roberta_L23_ridge1000.pickle (RoBERTa-large, layer 23 = -2,
# mean word/msg aggregation, ridge alpha=1000, 1024 features named 0me..1023me).
#
# Pipeline:
#   1. sentence-tokenize responses (creates personallm.responses_stoks)
#   2. build phq_outcomes (one row per message_id, PHQtot=NULL) so DLATK knows
#      which group ids to score
#   3. extract RoBERTa-large second-to-last-layer embeddings at message level
#      -> feat$roberta_la_meL23con$responses$message_id
#   4. apply the pickle, writing predictions to personallm.dep_text
#
# Run from /data/personaLLM so relative paths resolve.
set -euo pipefail

cd "$(dirname "$0")/../dlatk"

PY=/home/ssubrahmanya/.conda/envs/dlatk/bin/python
ROOT=/data/personaLLM

# 1. Sentence-tokenize -> personallm.responses_stoks (skip the WARNING path
#    inside addEmbTable that tokenizes on the fly every run).
$PY dlatkInterface.py \
    -d personallm \
    -t responses \
    -c message_id \
    --add_sent_tokenized

# 2. Outcome table (message_id + placeholder PHQtot) for DLATK's apply step.
cd "$ROOT"
uv run --quiet python src/prepare_phq_outcomes.py
cd "$ROOT/dlatk"

# 3. RoBERTa-large embeddings, second-to-last layer (L23), mean word-pooled
#    per message. Produces feat$roberta_la_meL23con$responses$message_id with
#    1024 feats (0me..1023me) matching the pickle's featureNamesList.
$PY dlatkInterface.py \
    -d personallm \
    -t responses \
    -c message_id \
    --add_emb_feat \
    --emb_model roberta-large \
    --emb_layers -2

# 4. Apply the pretrained ridge pickle -> personallm.dep_text
#    (column: PHQtot_dep_text with per-message_id predictions)
$PY dlatkInterface.py \
    -d personallm \
    -t responses \
    -c message_id \
    -f 'feat$roberta_la_meL23con$responses$message_id' \
    --outcome_table phq_outcomes \
    --outcomes PHQtot \
    --predict_regression_to_outcome_table dep_text \
    --load \
    --picklefile "$ROOT/models/dep_text_PHQ_roberta_L23_ridge1000.pickle"
