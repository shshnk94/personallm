"""Build personallm.phq_outcomes: one row per responses.message_id with a
placeholder PHQtot column.

DLATK's --predict_regression_to_outcome_table needs an outcome table to know
which group ids to score; the column itself is not used when --load is set.
"""

import pandas as pd
from sqlalchemy import create_engine, text


if __name__ == "__main__":
    engine = create_engine("mysql+pymysql://ssubrahmanya@/personallm?charset=utf8mb4")

    ids = pd.read_sql("SELECT message_id FROM responses", engine)
    ids["PHQtot"] = pd.NA
    ids = ids.astype({"message_id": "int64", "PHQtot": "Float64"})

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS phq_outcomes"))
        ids.to_sql("phq_outcomes", conn, index=False)
        conn.execute(text("ALTER TABLE phq_outcomes ADD PRIMARY KEY (message_id)"))

    print(f"Prepared phq_outcomes with {len(ids)} rows")
