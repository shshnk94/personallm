"""Load a full lead.csv export into a MySQL/MariaDB table.

Loads every column in the CSV (not just the PHQ-9/GAD-7/PSS-10 scale
columns). If the CSV lacks a `message_id` column one is generated from the
row index, and it is used as the primary key.

Column names containing the Qualtrics `Field[Code]` bracket syntax are
flattened so they're safe as plain MySQL identifiers: the three scale
item/total columns keep their established names (e.g. `PHQ[PHQ1]` ->
`PHQ_item1`, `PHQtot` -> `phq9`) so they line up with any pre-existing table
of the same name, and the free-text response column (`GenText`) is renamed
to `message` so DLATK can pick it up by default. Every other bracketed
column (e.g. `Diagnosis[NONE]`) is flattened generically to `Diagnosis__NONE`
(double underscore, so it can't collide with a pre-existing plain-underscore
column such as `employment_other`).

If `--table` already exists, columns whose (post-rename) name matches an
existing column reuse that column's current SQL type -- this is what lets
you overwrite a table with a richer CSV export without changing the dtype of
columns other code already depends on. Columns with no existing counterpart
get a type inferred from the data (pandas/SQLAlchemy's normal to_sql
inference), except columns that look like timestamps, which are parsed to
datetime first so they land as DATETIME instead of TEXT.

Connects to the local MariaDB instance via its unix socket:

    uv run python load_csv.py \\
        --csv data/lead.csv --table lead --database ssubrahmanya
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.types import BIGINT, BOOLEAN, DATETIME, DOUBLE, FLOAT, INTEGER, TEXT, VARCHAR

# Source text column; renamed to `message` so DLATK can pick it up by default.
TEXT_COL = "GenText"

# Items go to `<SCALE>_item<N>` (e.g. PHQ[PHQ1] -> PHQ_item1) and totals to
# the scale name (e.g. PHQtot -> phq9). The `_item` suffix on items keeps
# them from colliding with the totals, which MySQL treats case-insensitively.
_ITEM_RE = re.compile(r"^(PHQ|GAD|PSS)\[(?:PHQ|GAD|PSS)0*(\d+)\]$")
_TOTAL_RENAMES = {"PHQtot": "phq9", "GADtot": "gad7", "PSStot": "pss10"}

# Any other column still containing Qualtrics' `Field[Code]` syntax after the
# renames above (e.g. `Diagnosis[NONE]`, `employment[1]`) is flattened with a
# double underscore so it can't collide with an existing plain-underscore
# column name.
_BRACKET_RE = re.compile(r"[\[\]]")

# Column names matching this (case-insensitive) are parsed as timestamps if
# the values actually parse, so they land in MySQL as DATETIME rather than
# generic TEXT.
_DATE_LIKE_RE = re.compile(r"date|timestamp", re.IGNORECASE)


def _rename(c: str) -> str:
    if c == TEXT_COL:
        return "message"
    if c in _TOTAL_RENAMES:
        return _TOTAL_RENAMES[c]
    m = _ITEM_RE.match(c)
    if m:
        return f"{m.group(1)}_item{int(m.group(2))}"
    return _BRACKET_RE.sub(lambda mo: "__" if mo.group() == "[" else "", c)


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object and _DATE_LIKE_RE.search(col):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() == df[col].notna().sum():
                df[col] = parsed
    return df


# Coarse MySQL type-string -> SQLAlchemy type mapping, covering the types
# already present on tables in this database. Falls back to `None` (meaning:
# let pandas infer) for anything unrecognized.
def _sa_type_for(mysql_type: str):
    t = mysql_type.lower()
    if t.startswith("bigint"):
        return BIGINT()
    if t.startswith(("int", "smallint", "mediumint")):
        return INTEGER()
    if t.startswith("double"):
        return DOUBLE()
    if t.startswith("float"):
        return FLOAT()
    if t.startswith("tinyint(1)"):
        return BOOLEAN()
    if t.startswith("datetime") or t.startswith("timestamp"):
        return DATETIME()
    if t.startswith("text"):
        return TEXT()
    m = re.match(r"varchar\((\d+)\)", t)
    if m:
        return VARCHAR(int(m.group(1)))
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    parser.add_argument("--table", required=True, help="MySQL table name")
    parser.add_argument("--database", required=True, help="MySQL database name")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)

    df = df.rename(columns=_rename)
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        raise SystemExit(f"Column rename produced duplicate names: {dupes}")

    if "message_id" not in df.columns:
        df = df.reset_index(names="message_id")
    df = df.astype({"message_id": "int64"})
    df = _parse_dates(df)

    engine = create_engine(
        f"mysql://ssubrahmanya@/{args.database}?charset=utf8mb4",
        connect_args={"read_default_file": str(Path.home() / ".my.cnf")},
    )

    dtype_overrides = {}
    if inspect(engine).has_table(args.table):
        for col in inspect(engine).get_columns(args.table):
            if col["name"] in df.columns:
                sa_type = _sa_type_for(str(col["type"]))
                if sa_type is not None:
                    dtype_overrides[col["name"]] = sa_type

    with engine.connect() as conn:
        df.to_sql(
            name=args.table,
            con=conn,
            if_exists="replace",
            index=False,
            dtype=dtype_overrides or None,
        )
        conn.execute(text(f"ALTER TABLE {args.table} ADD PRIMARY KEY (message_id)"))

    print(f"Loaded {len(df)} rows x {df.shape[1]} cols into {args.database}.{args.table}")
    if dtype_overrides:
        print(f"Reused existing dtypes for {len(dtype_overrides)} matching columns: {sorted(dtype_overrides)}")
