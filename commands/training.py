import argparse
import subprocess
from pathlib import Path

### ---- Configuration ----

DATABASE = "ssubrahmanya"
MESSAGE_TABLE = "lead_en_train"
OUTCOME_TABLE = "lead_en_train"
OUTCOME = "phq9"

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
    args = parser.parse_args()

    # 1. Embedding extraction, mean word-pooled per message.
    if args.extract_embeddings:
        run(
            [
                PYTHON, "dlatk/dlatkInterface.py",
                "-d", DATABASE,
                "-t", args.message_table,
                "-c", "message_id",
                "--add_emb_feat",
                "--embedding_model", EMBEDDING_MODEL,
                "--embedding_word_aggregation", EMBEDDING_WORD_AGGREGATION,
                "--embedding_layers", EMBEDDING_LAYERS,
            ]
        )

    feature_table = (
        f"feat${EMBEDDING_MODEL_SHORT}_{word_agg}L{EMBEDDING_LAYERS}con"
        f"${args.message_table}$message_id"
    )
    pickle_file = (
        Path("results") /
        f"genText_{args.outcome}_{EMBEDDING_MODEL_SHORT}_L{EMBEDDING_LAYERS}_ridge{ALPHA}.pkl"
    )

    # 2. 10-fold cross-validated ridge regression (alpha=1000) predicting the
    #    outcome, plus a final model trained on all data saved to results/.
    if args.train:
        run(
            [
                PYTHON, "dlatk/dlatkInterface.py",
                "-d", DATABASE,
                "-t", args.message_table,
                "-c", "message_id",
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

    # 3. Apply the saved model and write per-message predictions to
    #    feat$p_ridg_{outcome}_pred${message_table}$message_id.
    if args.extract:
        run(
            [
                PYTHON, "dlatk/dlatkInterface.py",
                "-d", DATABASE,
                "-t", args.message_table,
                "-c", "message_id",
                "-f", feature_table,
                "--outcome_table", args.outcome_table,
                "--outcomes", args.outcome,
                "--group_freq_thresh", "0",
                "--predict_regression_to_feats", f"{args.outcome}_pred",
                "--load",
                "--picklefile", pickle_file,
            ]
        )