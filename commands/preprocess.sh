#!/usr/bin/env bash
# Retain only English-language messages from ssubrahmanya.lead, producing
# ssubrahmanya.lead_en (DLATK creates it with `CREATE TABLE lead_en LIKE lead`
# and copies the rows that langid classifies as English).
#
# Reference: https://dlatk.github.io/dlatk/tutorials/tut_data_cleaning.html
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/python dlatk/dlatkInterface.py \
    -d ssubrahmanya \
    -t lead \
    -c message_id \
    --language_filter en \
    --clean_messages
