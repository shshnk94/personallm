from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from sqlalchemy import create_engine, text


def slug(name: str) -> str:
    name = name.rsplit("/", 1)[-1]
    return re.sub(r"[^0-9a-zA-Z]+", "", name).lower()


MESSAGE_TABLES = {
    "human": "lead_en_valid",
    "phq9": {
        "perItems": "perItemsPHQ9",
        "demoItems": "demoItemsPHQ9",
        "noPerItems": "noPerItemsPHQ9",
        "perSever": "perSeverPHQ9",
    },
    "gad7": {
        "perItems": "perItemsGAD7",
        "demoItems": "demoItemsGAD7",
        "noPerItems": "noPerItemsGAD7",
    },
    "pss10": {
        "perItems": "perItemsPSS10",
        "demoItems": "demoItemsPSS10",
        "noPerItems": "noPerItemsPSS10",
    },
}


TRIAL = 0


def message_table_for(source: str, scale: str) -> str:
    if source == "human":
        return MESSAGE_TABLES["human"]
    if source in MESSAGE_TABLES[scale]:
        return f"{MESSAGE_TABLES[scale][source]}_t{TRIAL}"
    raise SystemExit(f"unknown --source {source!r}; expected 'human' or one of {sorted(MESSAGE_TABLES[scale])}")


repo_root = Path(__file__).resolve().parent.parent


def build_table(database: str, source: str, scale: str) -> None:
    table_name = message_table_for(source, scale)

    engine = create_engine(
        f"mysql://ssubrahmanya@/{database}?charset=utf8mb4",
        connect_args={
            "read_default_file": str(Path.home() / ".my.cnf")
        },
    )

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        if source == "human":
            conn.execute(text(f'''
                CREATE TABLE {table_name} AS
                SELECT *, ProlificID AS participant_id
                FROM lead_en
                WHERE ProlificID IN (SELECT participant_id FROM lead_outcomes_agg_valid)
            '''))
            conn.execute(text(f"ALTER TABLE {table_name} MODIFY participant_id VARCHAR(40) NOT NULL"))
        else:
            base_table = MESSAGE_TABLES[scale][source]
            conn.execute(
                text(f"CREATE TABLE {table_name} AS SELECT * FROM {base_table} WHERE trial = :trial"),
                {"trial": TRIAL},
            )
        conn.execute(text(f"ALTER TABLE {table_name} ADD PRIMARY KEY (message_id)"))
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()

    print(f"[{table_name}: {count} rows for source={source}]")


def feature_extraction(
    database: str,
    source: str,
    scale: str,
    group_freq_thresh: int,
    set_p_occ: bool = False,
    occ: float = 0.05,
    lexicon: str = "LIWC2015",
) -> None:
    message_table = message_table_for(source, scale)
    grouping_field = "participant_id"

    subprocess.run(
        [str(c) for c in [
            ".venv/bin/python3", "dlatk/dlatkInterface.py",
            "-d", database,
            "-t", message_table,
            "-c", grouping_field,
            "--add_ngrams", "-n", "1"
        ]],
        cwd=repo_root, check=True,
    )

    if set_p_occ:
        ngram_feat_table = f"feat$1gram${message_table}${grouping_field}"
        subprocess.run(
            [str(c) for c in [
                ".venv/bin/python3", "dlatk/dlatkInterface.py",
                "-d", database,
                "-t", message_table,
                "-c", grouping_field,
                "--group_freq_thresh", str(group_freq_thresh),
                "-f", ngram_feat_table,
                "--feat_occ_filter", "--set_p_occ", str(occ),
            ]],
            cwd=repo_root, check=True,
        )

    subprocess.run(
        [str(c) for c in [
            ".venv/bin/python3", "dlatk/dlatkInterface.py",
            "-d", database,
            "-t", message_table,
            "-c", grouping_field,
            "--add_lex_table", "-l", lexicon,
            "--weighted_lexicon"
        ]],
        cwd=repo_root, check=True,
    )


def correlate(
    database: str,
    source: str,
    scale: str,
    group_freq_thresh: int,
    feature_table: str,
    output_name: str,
) -> None:
    message_table = message_table_for(source, scale)
    grouping_field = "participant_id"
    outcome_table = message_table
    outcomes = [scale]

    subprocess.run(
        [str(c) for c in [
            ".venv/bin/python3", "dlatk/dlatkInterface.py",
            "-d", database,
            "-t", message_table,
            "-c", grouping_field,
            "-f", feature_table,
            "--outcome_table", outcome_table, "--outcomes", *outcomes,
            "--group_freq_thresh", str(group_freq_thresh),
            "--output_name", output_name,
            "--correlate", "--csv",
            "--tagcloud", "--make_wordclouds"
        ]],
        cwd=repo_root, check=True,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", default="ssubrahmanya", help="MySQL database holding the message/outcome tables (default: %(default)s)")
    parser.add_argument("--scale", default="phq9", help="Which scale's response/outcome table to run DLA over.")
    parser.add_argument("--source", default="human", choices=["human", "perItems", "demoItems", "noPerItems", "perSever"], help="'human' for the human validation split, or a prompting strategy from the perItems/demoItems/noPerItems/perSever vignette tables")

    parser.add_argument("--lexicon", default="LIWC2015", help="Lexicon table (in DLATK's dlatk_lexica database) for the LIWC feature table (default: %(default)s)")
    parser.add_argument("--set-p-occ", action="store_true", help="Apply --occ's person-occurrence filter to the 1gram feature table and correlate that filtered table (default: correlate the unfiltered 1gram table)")
    parser.add_argument("--occ", type=float, default=0.05, help="Minimum fraction of persons an n-gram must occur in to survive --feat_occ_filter; only used when --set-p-occ is given (default: %(default)s)")
    parser.add_argument("--group-freq-thresh", type=int, default=0, help="DLATK's minimum word count per person to be kept, applied during filtering/correlation (default: %(default)s)")
    parser.add_argument("--output-dir", default="results/dla", help="Where --correlate writes its rMatrix/tagcloud output (default: %(default)s)")
    parser.add_argument("--extract", action="store_true", help="Step 1: build the 1gram (occurrence-filtered) and LIWC feature tables")
    parser.add_argument("--correlate", action="store_true", help="Step 2: correlate both feature tables with --outcomes")
    args = parser.parse_args()

    message_table = message_table_for(args.source, args.scale)
    group_field = "participant_id"

    if not (args.extract or args.correlate):
        args.extract = args.correlate = True

    ngram_feat_table = f"feat$1gram${message_table}${group_field}"
    if args.set_p_occ:
        ngram_feat_table = f"{ngram_feat_table}${str(args.occ).replace('.', '_')}"
    meta_ngram_feat_table = f"feat$meta_1gram${message_table}${group_field}"
    liwc_feat_table = f"feat$cat_{args.lexicon}_w${message_table}${group_field}$1gra"

    if args.extract:
        build_table(args.database, args.source, args.scale)

        feature_extraction(
            args.database,
            args.source,
            args.scale,
            args.group_freq_thresh,
            args.set_p_occ,
            args.occ,
            args.lexicon,
        )

    if args.correlate:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"dla_{args.scale}_{slug(args.source)}"

        correlate(
            args.database,
            args.source,
            args.scale,
            args.group_freq_thresh,
            ngram_feat_table,
            f"{args.output_dir}/{stem}_1gram",
        )

        correlate(
            args.database,
            args.source,
            args.scale,
            args.group_freq_thresh,
            meta_ngram_feat_table,
            f"{args.output_dir}/{stem}_meta_1gram",
        )

        correlate(
            args.database,
            args.source,
            args.scale,
            args.group_freq_thresh,
            liwc_feat_table,
            f"{args.output_dir}/{stem}_liwc",
        )
