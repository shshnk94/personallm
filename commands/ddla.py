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
    raise SystemExit(f"unknown --source {source!r}; expected one of {sorted(MESSAGE_TABLES[scale])}")


repo_root = Path(__file__).resolve().parent.parent


def build_ddla_table(database: str, source: str, scale: str) -> None:
    human_table = message_table_for("human", scale)
    model_table = message_table_for(source, scale)
    ddla_table = f"ddla_{scale}_{slug(source)}"

    engine = create_engine(
        f"mysql://ssubrahmanya@/{database}?charset=utf8mb4",
        connect_args={"read_default_file": str(Path.home() / ".my.cnf")},
    )

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {ddla_table}"))
        conn.execute(text(f"""
            CREATE TABLE {ddla_table} AS
            SELECT CONCAT('0_', participant_id) AS ddla_group_id, message, {scale}, 0 AS is_model
            FROM {human_table}
            UNION ALL
            SELECT CONCAT('1_', participant_id) AS ddla_group_id, message, {scale}, 1 AS is_model
            FROM {model_table}
        """))
        conn.execute(text(f"ALTER TABLE {ddla_table} ADD COLUMN message_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST"))
        conn.execute(text(f"ALTER TABLE {ddla_table} ADD INDEX (ddla_group_id)"))
        counts = dict(conn.execute(
            text(f"SELECT is_model, COUNT(*) FROM {ddla_table} GROUP BY is_model")
        ).fetchall())

    print(f"[{ddla_table}: {counts.get(0, 0)} human + {counts.get(1, 0)} model rows]")


def feature_extraction(
    database: str,
    source: str,
    scale: str,
    group_freq_thresh: int,
    set_p_occ: bool = False,
    occ: float = 0.05,
    lexicon: str = "LIWC2015",
) -> None:
    ddla_table = f"ddla_{scale}_{slug(source)}"
    grouping_field = "ddla_group_id"

    subprocess.run(
        [str(c) for c in [
            ".venv/bin/python3", "dlatk/dlatkInterface.py",
            "-d", database,
            "-t", ddla_table,
            "-c", grouping_field,
            "--add_ngrams", "-n", "1"
        ]],
        cwd=repo_root, check=True,
    )

    if set_p_occ:
        ngram_feat_table = f"feat$1gram${ddla_table}${grouping_field}"
        subprocess.run(
            [str(c) for c in [
                ".venv/bin/python3", "dlatk/dlatkInterface.py",
                "-d", database,
                "-t", ddla_table,
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
            "-t", ddla_table,
            "-c", grouping_field,
            "--add_lex_table", "-l", lexicon,
            "--weighted_lexicon"
        ]],
        cwd=repo_root, check=True,
    )


def ddla(
    database: str,
    source: str,
    scale: str,
    group_freq_thresh: int,
    ddla_pvalue: float,
    feature_table: str,
    output_name: str,
) -> None:
    ddla_table = f"ddla_{scale}_{slug(source)}"
    grouping_field = "ddla_group_id"
    outcome_table = ddla_table
    outcome = scale

    subprocess.run(
        [str(c) for c in [
            ".venv/bin/python3", "dlatk/dlatkInterface.py",
            "-d", database,
            "-t", ddla_table,
            "-c", grouping_field,
            "--group_freq_thresh", str(group_freq_thresh),
            "-f", feature_table,
            "--outcome_table", outcome_table, "--outcomes", outcome,
            "--interaction_ddla", "is_model",
            "--interaction_ddla_pvalue", str(ddla_pvalue),
            "--output_interaction_terms",
            "--output_name", output_name,
            "--csv", "--correlate",
            "--tagcloud", "--make_wordclouds"
        ]],
        cwd=repo_root, check=True,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", default="ssubrahmanya", help="MySQL database holding the message/outcome tables (default: %(default)s)")
    parser.add_argument("--scale", required=True, choices=["phq9", "gad7", "pss10"], help="Which scale's response/outcome table to run DDLA over.")
    parser.add_argument("--source", required=True, choices=["perItems", "demoItems", "noPerItems"], help="Prompting strategy to compare against human text, from the perItems/demoItems/noPerItems vignette tables")

    parser.add_argument("--lexicon", choices=["LIWC2007", "LIWC2015", "LIWC2022"], default="LIWC2015", help="Lexicon table (in DLATK's dlatk_lexica database) for the LIWC feature table (default: %(default)s)")
    parser.add_argument("--set-p-occ", action="store_true", help="Apply --occ's person-occurrence filter to the 1gram feature table and correlate that filtered table (default: correlate the unfiltered 1gram table)")
    parser.add_argument("--occ", type=float, default=0.05, help="Minimum fraction of persons an n-gram must occur in to survive --feat_occ_filter; only used when --set-p-occ is given (default: %(default)s)")
    parser.add_argument("--group-freq-thresh", type=int, default=0, help="DLATK's minimum word count per person to be kept, applied during filtering/correlation (default: %(default)s)")
    parser.add_argument("--output-dir", default="results/dla", help="Where --correlate writes its rMatrix/tagcloud output (default: %(default)s)")
    parser.add_argument("--ddla-pvalue", type=float, default=0.001, help="Significance threshold for a feature's outcome x is_model interaction term to be re-tested (default: %(default)s)")
    args = parser.parse_args()

    build_ddla_table(args.database, args.source, args.scale)
    feature_extraction(
        args.database,
        args.source,
        args.scale,
        args.group_freq_thresh,
        args.set_p_occ,
        args.occ,
        args.lexicon,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ddla_{args.scale}_{slug(args.source)}"

    ddla_table = f"ddla_{args.scale}_{slug(args.source)}"
    grouping_field = "ddla_group_id"
    ngram_feat_table = f"feat$1gram${ddla_table}${grouping_field}"
    if args.set_p_occ:
        ngram_feat_table = f"{ngram_feat_table}${str(args.occ).replace('.', '_')}"
    liwc_feat_table = f"feat$cat_{args.lexicon}_w${ddla_table}${grouping_field}$1gra"

    ddla(
        args.database,
        args.source,
        args.scale,
        args.group_freq_thresh,
        args.ddla_pvalue,
        ngram_feat_table,
        f"{args.output_dir}/{stem}_1gram",
    )

    ddla(
        args.database,
        args.source,
        args.scale,
        args.group_freq_thresh,
        args.ddla_pvalue,
        liwc_feat_table,
        f"{args.output_dir}/{stem}_liwc",
    )
