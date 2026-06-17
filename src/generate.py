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
from .prompts.vignette import (
    GAD7_VIGNETTE,
    GENDER_NOUNS,
    MAX_SCORE,
    NAMES,
    PHQ9_VIGNETTE,
    PRONOUNS,
    PSS10_VIGNETTE,
    REVERSE,
    VERBS,
    symptom_narrative,
)

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
        "vignette": PHQ9_VIGNETTE,
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
        "vignette": GAD7_VIGNETTE,
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
        "vignette": PSS10_VIGNETTE,
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

# Severity bands per scale as (low, high, label); labels are lower-cased to read
# naturally in the vignette ("presents with {label} depression").
BANDS = {
    "phq9": [
        (0, 0, "no"),
        (1, 4, "minimal"),
        (5, 9, "mild"),
        (10, 14, "moderate"),
        (15, 19, "moderately severe"),
        (20, 27, "severe"),
    ],
    "gad7": [
        (0, 0, "no"),
        (1, 4, "minimal"),
        (5, 9, "mild"),
        (10, 14, "moderate"),
        (15, 21, "severe"),
    ],
    "pss10": [
        (0, 0, "no"),
        (1, 13, "low"),
        (14, 26, "moderate"),
        (27, 40, "high"),
    ],
}

# Band-conditioned impairment sentence for the vignette. Non-empty values carry
# a leading space so they slot in cleanly; "" omits the sentence entirely.
IMPAIRMENT = {
    "phq9": {
        "no": "",
        "minimal": "",
        "mild": " Functioning is mildly affected.",
        "moderate": " Functioning is moderately impaired.",
        "moderately severe": " Functioning is significantly impaired.",
        "severe": " Functioning is severely impaired.",
    },
    "gad7": {
        "no": "",
        "minimal": "",
        "mild": " Functioning is mildly affected.",
        "moderate": " Functioning is moderately impaired.",
        "severe": " Functioning is severely impaired.",
    },
    "pss10": {
        "no": "",
        "low": "",
        "moderate": " Coping is moderately strained.",
        "high": " Coping is severely overwhelmed.",
    },
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def build_prompt(
    variant: str,
    scale_name: str,
    scale: dict,
    item_scores: list[int],
    gender: str | None = None,
    age: int | None = None,
) -> str:
    """Build the prompt for a record under the chosen variant.

    'structured' renders the JSON per-item template (prompts/prompt.py);
    'vignette' renders a persona narrative (prompts/vignette.py) and needs a
    normalized `gender` category ("female"/"male"/"neutral") and `age`.
    """
    items = scale["items"]

    if variant == "vignette":
        total = total_score(scale_name, items, item_scores)
        label = severity_label(scale_name, total)
        narrative = symptom_narrative(scale_name, items, item_scores) or "no specific symptoms"
        return scale["vignette"].format(
            name=NAMES[gender],
            age=age,
            gender=GENDER_NOUNS[gender],
            pronoun=PRONOUNS[gender],
            verb=VERBS[gender],
            severity_label=label,
            score=total,
            symptom_narrative=narrative,
            functioning=IMPAIRMENT[scale_name][label],
            question=QUESTION,
        )

    payload = {
        "item_scores": [
            {"item": item, "score": int(score)}
            for item, score in zip(items, item_scores)
        ],
        "question": QUESTION,
    }
    return scale["template"].format(input=json.dumps(payload, indent=2))


def total_score(scale_name: str, items: list[str], item_scores: list[int]) -> int:
    """Sum item scores; reverse-scored items are inverted first (PSS-10)."""
    reverse, max_score = REVERSE[scale_name], MAX_SCORE[scale_name]
    return sum(
        (max_score - int(s)) if item in reverse else int(s)
        for item, s in zip(items, item_scores)
    )


def severity_label(scale_name: str, total: int) -> str:
    for low, high, label in BANDS[scale_name]:
        if low <= total <= high:
            return label
    return BANDS[scale_name][-1][2]


# Raw lead.csv Gender codes -> persona category; anything else (GVNC, -oth-,
# NotSa, TM, TF, ...) falls back to a neutral they/them persona.
GENDER_CATEGORY = {"F": "female", "M": "male"}
DEFAULT_AGE = 40


def gender_category(raw: str) -> str:
    return GENDER_CATEGORY.get(str(raw).strip(), "neutral")


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

    # Per-item scores come from the source table (`a`); Age/Gender live only in
    # the raw `lead` table (`b`). Inner-joining on message_id filters to the
    # source rows while pulling demographics across, so a score split such as
    # lead_en_valid (which lacks demographics) can still drive the vignette.
    select = ", ".join([f"a.{c}" for c in ["message_id"] + columns] + ["b.Age", "b.Gender"])
    query = f"SELECT {select} FROM {table} a, lead b WHERE a.message_id = b.message_id"
    df = pd.read_sql(query, engine)
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate responses for LLM benchmarking.")
    parser.add_argument("--scale", choices=sorted(SCALES.keys()), required=True, help="Which scale to prompt with.")
    parser.add_argument("--prompt-variant", choices=["structured", "vignette"], default="structured", help="Prompt family: 'structured' (prompts/prompt.py) or 'vignette' (prompts/vignette.py).")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Model names to benchmark (provider-qualified for openrouter, e.g. openai/gpt-4.1-mini).")
    parser.add_argument("--source-table", default="lead_en", help="MySQL table to pull per-item scores from.")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=6, help="Max retry attempts per request on 429/5xx/timeouts.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    # setting up necessary logging
    setup_logging()

    # load the API tokens from the .env file
    load_dotenv()
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY is not set (add to environment or .env)")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scale = SCALES[args.scale]
    sources = get_source_labels(args.source_table, scale["columns"])
    logging.info(f"loaded {len(sources)} rows from {args.source_table} for {args.scale}")

    # The vignette variant personalizes each record with its own Age/Gender
    # (pulled from the source table by get_source_labels); structured has none.
    if args.prompt_variant == "vignette":
        missing = int(sources["Gender"].isna().sum())
        if missing:
            logging.warning("%d rows missing demographics; using neutral persona / age %d", missing, DEFAULT_AGE)
        cats = sources["Gender"].map(gender_category)
        logging.info("persona categories: %s", cats.value_counts().to_dict())

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )

    rows = []
    for model in args.models:

        logging.info(f"generating responses for model {model}...")
        for idx, record in tqdm(sources.iterrows(), total=len(sources), desc=f"{args.scale}/{args.prompt_variant}"):

            item_scores = [int(record[col]) for col in scale["columns"]]

            if args.prompt_variant == "vignette":
                gender = gender_category(record["Gender"])
                age = int(record["Age"]) if pd.notna(record["Age"]) else DEFAULT_AGE
            else:
                gender, age = None, None
            prompt = build_prompt(
                args.prompt_variant, args.scale, scale, item_scores, gender, age
            )

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
                    "variant": args.prompt_variant,
                    "gender": gender,
                    "age": age,
                    "item_scores": json.dumps(item_scores),
                    "text": text,
                    "error": err,
                }
            )

    suffix = "" if args.prompt_variant == "structured" else f"_{args.prompt_variant}"
    output_path = args.output_dir / f"generated_responses_{args.scale}{suffix}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logging.info("wrote %s", output_path)