import argparse
import subprocess
from pathlib import Path

from sqlalchemy import create_engine, text

### ---- Configuration ----

DATABASE = "ssubrahmanya"
MESSAGE_TABLE = "lead_en_train"
OUTCOME_TABLE = "lead_en_train"
OUTCOME = "phq9"
CORREL_FIELD = "message_id"
TRIAL_FIELD = "trial"

EMBEDDING_MODEL = "roberta-large"
EMBEDDING_WORD_AGGREGATION = "mean"
EMBEDDING_LAYERS = "23"

ALPHA = 1000
NFOLDS = 10

# Pin to the project's .venv so it picks the right interpreter
PYTHON = ".venv/bin/python3"

# Derive DLATK's short model name (see dlatk/dlatk/featureExtractor.py addEmbTable)
# strip any "org/" prefix, split on "-", keep the first piece whole, 
# truncate each remaining piece to 2 chars, join with "_".
# e.g. roberta-large -> roberta_la, bert-base-uncased -> bert_ba_un.
_pieces = EMBEDDING_MODEL.rsplit(sep="/", maxsplit=1)[-1].split("-")
EMBEDDING_MODEL_SHORT = "_".join([_pieces[0]] + [p[:2] for p in _pieces[1:]])

# The "con" suffix is layerAggregations="concatenate" (first 2 chars) plus the
# literal "n" DLATK always appends; if you ever pass a different
# --embedding_layer_aggregation, update that segment.
word_agg = EMBEDDING_WORD_AGGREGATION[:2]
### ---- Execution ----

def run(command):
    repo_root = Path(__file__).resolve().parent.parent
    subprocess.run(
        command,
        cwd=repo_root,
        check=True
    )


def get_engine():
    return create_engine(
        f"mysql://ssubrahmanya@/{DATABASE}?charset=utf8mb4",
        connect_args={"read_default_file": str(Path.home() / ".my.cnf")},
    )


def extract_by_trial(args):
    """Apply the saved participant-level ridge model to args.message_table
    one trial at a time, stacking the results into one combined
    (participant_id, trial)-level prediction table.

    args.correl_field is ignored here -- the point of this step is a
    participant-level score per trial, so each trial's filtered table is
    always grouped on participant_id.
    """
    correl_field = "participant_id"
    feature_table_prefix = f"feat${EMBEDDING_MODEL_SHORT}_{word_agg}L{EMBEDDING_LAYERS}con"
    pickle_file = (
        Path("results") /
        f"genText_{args.outcome}_{correl_field}_{EMBEDDING_MODEL_SHORT}_L{EMBEDDING_LAYERS}_ridge{ALPHA}.pkl"
    )

    engine = get_engine()
    with engine.connect() as conn:
        trials = [
            row[0] for row in conn.execute(
                text(f"SELECT DISTINCT {TRIAL_FIELD} FROM {args.message_table} ORDER BY {TRIAL_FIELD}")
            )
        ]

    combined_table = f"feat$p_ridg_{args.outcome}_pred${args.message_table}$participant_id_trial"
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS `{combined_table}`"))
        conn.execute(text(
            f"CREATE TABLE `{combined_table}` ("
            "participant_id VARCHAR(40) NOT NULL, "
            f"{TRIAL_FIELD} BIGINT NOT NULL, "
            "feat VARCHAR(20) NOT NULL, "
            "value DOUBLE, group_norm DOUBLE, "
            f"PRIMARY KEY (participant_id, {TRIAL_FIELD}))"
        ))

    for trial in trials:
        # "_t{trial}" rather than "_trial{trial}" -- DLATK's feat$ table
        # names stack a ~40-char prefix/suffix around this, and the longer
        # suffix pushed feat$roberta_la_meL23con$..._trial0$participant_id
        # past MySQL's 64-char identifier limit (error 1103).
        temp_table = f"{args.message_table}_t{trial}"
        emb_feature_table = f"{feature_table_prefix}${temp_table}${correl_field}"
        pred_feature_table = f"feat$p_ridg_{args.outcome}_pred${temp_table}${correl_field}"

        try:
            with engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS `{temp_table}`"))
                conn.execute(
                    text(
                        f"CREATE TABLE `{temp_table}` AS "
                        f"SELECT * FROM `{args.message_table}` WHERE {TRIAL_FIELD} = :trial"
                    ),
                    {"trial": trial},
                )
                conn.execute(text(f"ALTER TABLE `{temp_table}` ADD PRIMARY KEY (message_id)"))

            run(
                [
                    PYTHON, "dlatk/dlatkInterface.py",
                    "-d", DATABASE,
                    "-t", temp_table,
                    "-c", correl_field,
                    "--add_emb_feat",
                    "--embedding_model", EMBEDDING_MODEL,
                    "--embedding_word_aggregation", EMBEDDING_WORD_AGGREGATION,
                    "--embedding_layers", EMBEDDING_LAYERS,
                ]
            )

            run(
                [
                    PYTHON, "dlatk/dlatkInterface.py",
                    "-d", DATABASE,
                    "-t", temp_table,
                    "-c", correl_field,
                    "-f", emb_feature_table,
                    "--outcome_table", args.outcome_table,
                    "--outcomes", args.outcome,
                    "--group_freq_thresh", "0",
                    "--predict_regression_to_feats", f"{args.outcome}_pred",
                    "--load",
                    "--picklefile", pickle_file,
                ]
            )

            with engine.begin() as conn:
                conn.execute(text(
                    f"INSERT INTO `{combined_table}` (participant_id, {TRIAL_FIELD}, feat, value, group_norm) "
                    f"SELECT group_id, :trial, feat, value, group_norm FROM `{pred_feature_table}`"
                ), {"trial": trial})

            print(f"[trial {trial}: merged into {combined_table}]")
        finally:
            with engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS `{temp_table}`"))
                conn.execute(text(f"DROP TABLE IF EXISTS `{emb_feature_table}`"))
                conn.execute(text(f"DROP TABLE IF EXISTS `{pred_feature_table}`"))
            print(f"[trial {trial}: scratch tables dropped]")

    print(f"[done: {combined_table}]")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)

    parser.add_argument(
        "-e", "--extract-embeddings",
        action="store_true",
        help="Step 1: embedding extraction, off by default"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Step 2: cross-validated ridge training, off by default",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Step 3: apply saved pickle and write per-message predictions, off by default",
    )
    parser.add_argument(
        "--extract-by-trial",
        action="store_true",
        help=(
            "Step 4: apply the saved participant-level pickle one trial at a "
            "time (requires a trial column in --message-table) and write a "
            "combined (participant_id, trial)-level prediction table, off by default"
        ),
    )
    parser.add_argument(
        "--message-table",
        default=MESSAGE_TABLE,
        help=f"Message table (default: {MESSAGE_TABLE})",
    )
    parser.add_argument(
        "--outcome-table",
        default=OUTCOME_TABLE,
        help=f"Outcome table (default: {OUTCOME_TABLE})",
    )
    parser.add_argument(
        "--outcome",
        default=OUTCOME,
        help=f"Outcome to predict (default: {OUTCOME})",
    )
    parser.add_argument(
        "--correl-field",
        default=CORREL_FIELD,
        help=(
            "Grouping field shared by the message table and the outcome "
            f"table -- message_id for per-message, participant_id for "
            f"per-person (default: {CORREL_FIELD})"
        ),
    )
    args = parser.parse_args()

    # 1. Embedding extraction, mean word-pooled per group (per message if
    #    --correl-field message_id, per participant if participant_id).
    if args.extract_embeddings:
        run(
            [
                PYTHON, "dlatk/dlatkInterface.py",
                "-d", DATABASE,
                "-t", args.message_table,
                "-c", args.correl_field,
                "--add_emb_feat",
                "--embedding_model", EMBEDDING_MODEL,
                "--embedding_word_aggregation", EMBEDDING_WORD_AGGREGATION,
                "--embedding_layers", EMBEDDING_LAYERS,
            ]
        )

    feature_table = (
        f"feat${EMBEDDING_MODEL_SHORT}_{word_agg}L{EMBEDDING_LAYERS}con"
        f"${args.message_table}${args.correl_field}"
    )
    pickle_file = (
        Path("results") /
        f"genText_{args.outcome}_{args.correl_field}_{EMBEDDING_MODEL_SHORT}_L{EMBEDDING_LAYERS}_ridge{ALPHA}.pkl"
    )

    # 2. 10-fold cross-validated ridge regression (alpha=1000) predicting the
    #    outcome, plus a final model trained on all data saved to results/.
    if args.train:
        run(
            [
                PYTHON, "dlatk/dlatkInterface.py",
                "-d", DATABASE,
                "-t", args.message_table,
                "-c", args.correl_field,
                "-f", feature_table,
                "--outcome_table", args.outcome_table,
                "--outcomes", args.outcome,
                "--nfold_regression",
                "--train_regression",
                "--model", f"ridge{ALPHA}",
                "--folds", str(NFOLDS),
                "--group_freq_thresh", "0",
                "--save",
                "--picklefile", pickle_file,
            ]
        )

    # 3. Apply the saved model and write per-group predictions to
    #    feat$p_ridg_{outcome}_pred${message_table}${correl_field}.
    if args.extract:
        run(
            [
                PYTHON, "dlatk/dlatkInterface.py",
                "-d", DATABASE,
                "-t", args.message_table,
                "-c", args.correl_field,
                "-f", feature_table,
                "--outcome_table", args.outcome_table,
                "--outcomes", args.outcome,
                "--group_freq_thresh", "0",
                "--predict_regression_to_feats", f"{args.outcome}_pred",
                "--load",
                "--picklefile", pickle_file,
            ]
        )

    # 4. Apply the saved participant-level model one trial at a time (each
    #    trial has exactly one message per participant, so grouping by
    #    participant_id within a single trial is just that trial's own
    #    prediction), then stack the per-trial results into one
    #    (participant_id, trial)-level table. Scratch tables -- the
    #    trial-filtered message table, its embedding feature table, and its
    #    prediction feature table -- are dropped once copied into the
    #    combined table.
    if args.extract_by_trial:
        extract_by_trial(args)