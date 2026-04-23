"""Load data/generated_prompts.csv into the personallm.responses table.

Connects to the local MariaDB instance via its unix socket as root, so this
script must be run with sudo:
    sudo -E uv run python load_csv.py
"""

import pandas as pd
from sqlalchemy import create_engine, text


if __name__ == "__main__":

    # read the CSV to confirm it looks right before loading
    df = pd.read_csv("data/generated_prompts.csv")
    df = df \
        .loc[:, ["model", "target_score", "text"]] \
        .reset_index(names="message_id") \
        .astype({"message_id": "int64", "target_score": "int64"}) \
        .rename(columns={"text": "message"})
    
    
    engine = create_engine("mysql://ssubrahmanya@/personallm?charset=utf8mb4")
    with engine.connect() as conn:
        df.to_sql(
            name="responses", 
            con=conn, 
            if_exists="replace", 
            index=False
        )
        conn.execute(text("ALTER TABLE responses ADD PRIMARY KEY (message_id)"))
        conn.execute(text("ALTER TABLE responses ADD INDEX model (model)"))

    print(f"Loaded {len(df)} rows into personallm.responses")