"""Calibration — stage 1: generate responses per (model, source row) for a scale.

Pulls per-item scores for the chosen scale (PHQ-9, GAD-7, or PSS-10) from a
MySQL table and asks each model to respond as someone with those scores, using
the matching prompt template in `src/prompts/prompt.py`. Writes
<output-dir>/generated_responses_<scale>.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from sqlalchemy import create_engine
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tqdm import tqdm

from .prompts.prompt import GAD7, PHQ9, PSS10

MODELS = ["openai/gpt-4.1-mini"]

RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

QUESTION = (
    "How is your mental health? Please describe how you have been over the last "
    "two weeks. You can, for example, write about your emotions, thoughts, "
    "behaviours, and/or symptoms related to your health."
)

SCALES = {
    "phq9": {
        "template": PHQ9,
        "items": [
            "Anhedonia",
            "Depressed Mood",
            "Insomnia or Hypersomnia",
            "Fatigue",
            "Poor Appetite or Overeating",
            "Worthlessness or Guilt",
            "Difficulty Concentrating",
            "Psychomotor Agitation or Retardation",
            "Suicidal Ideation",
        ],
        "columns": [f"PHQ_item{i}" for i in range(1, 10)],
    },
    "gad7": {
        "template": GAD7,
        "items": [
            "Nervousness",
            "Uncontrollable Worry",
            "Excessive Worry",
            "Trouble Relaxing",
            "Restlessness",
            "Irritability",
            "Apprehension",
        ],
        "columns": [f"GAD_item{i}" for i in range(1, 8)],
    },
    "pss10": {
        "template": PSS10,
        "items": [
            "Upset by Unexpected Events",
            "Lack of Control",
            "Nervous and Stressed",
            "Confidence in Coping (reverse)",
            "Things Going Your Way (reverse)",
            "Inability to Cope",
            "Control of Irritations (reverse)",
            "On Top of Things (reverse)",
            "Anger at Things Outside Control",
            "Overwhelming Difficulties",
        ],
        "columns": [f"PSS_item{i}" for i in range(1, 11)],
    },
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def build_prompt(template: str, items: list[str], item_scores: list[int]) -> str:
    payload = {
        "item_scores": [
            {"item": item, "score": int(score)}
            for item, score in zip(items, item_scores)
        ],
        "question": QUESTION,
    }
    return template.format(input=json.dumps(payload, indent=2))


def generate(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    max_attempts: int,
) -> tuple[str, str | None]:
    try:
        for attempt in Retrying(
            retry=retry_if_exception_type(RETRYABLE),
            wait=wait_exponential_jitter(initial=1, max=30),
            stop=stop_after_attempt(max_attempts),
            before_sleep=before_sleep_log(logging.getLogger(), logging.WARNING),
            reraise=True,
        ):
            with attempt:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort="none"
                )
        return resp.choices[0].message.content or "", None
    except Exception as e:
        return "", repr(e)


def get_source_labels(
    table: str, 
    columns: list[str]
) -> pd.DataFrame:

    engine = create_engine(
        "mysql://ssubrahmanya@/ssubrahmanya?charset=utf8mb4",
        connect_args={"read_default_file": str(Path.home() / ".my.cnf")},
    )

    columns = ", ".join(["message_id"] + columns)
    df = pd.read_sql(f"SELECT {columns} FROM {table}", engine)
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate responses for LLM benchmarking.")
    parser.add_argument("--scale", choices=sorted(SCALES.keys()), required=True, help="Which scale to prompt with.")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Model names to benchmark (provider-qualified for openrouter, e.g. openai/gpt-4.1-mini).")
    parser.add_argument("--source-table", default="lead_en", help="MySQL table to pull per-item scores from.")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=6, help="Max retry attempts per request on 429/5xx/timeouts.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    load_dotenv()
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY is not set (add to environment or .env)")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scale = SCALES[args.scale]
    sources = get_source_labels(args.source_table, scale["columns"])
    logging.info(f"loaded {len(sources)} rows from {args.source_table} for {args.scale}")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )

    rows = []
    for model in args.models:

        logging.info(f"generating responses for model {model}...")
        for idx, record in sources.iterrows():
            
            item_scores = [int(record[col]) for col in scale["columns"]]
            prompt = build_prompt(scale["template"], scale["items"], item_scores)
            text, err = generate(
                client,
                model,
                prompt,
                args.max_tokens,
                args.temperature,
                args.max_attempts,
            )
                
            rows.append(
                {
                    "message_id": int(record.message_id),
                    "model": model,
                    "scale": args.scale,
                    "item_scores": json.dumps(item_scores),
                    "text": text,
                    "error": err,
                }
            )

    output_path = args.output_dir / f"generated_responses_{args.scale}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logging.info("wrote %s", output_path)