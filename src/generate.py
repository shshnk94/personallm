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
from sqlalchemy import create_engine, inspect
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
    PHQ9_VIGNETTE,
    PSS10_VIGNETTE,
    children_phrase,
    clean_freetext,
    economy_earning_phrase,
    education_phrase,
    partner_phrase,
    persona_details,
    persona_intro,
    symptom_narrative_prose,
)
from .prompts.persona_items import (
    GAD7 as GAD7_PERSONA_ITEMS,
    PHQ9 as PHQ9_PERSONA_ITEMS,
    PSS10 as PSS10_PERSONA_ITEMS,
    total_score,
)
from .prompts.persona import (
    GAD7 as GAD7_PERSONA_TOTAL,
    PHQ9 as PHQ9_PERSONA_TOTAL,
    PSS10 as PSS10_PERSONA_TOTAL,
)

# Extra per-participant demographic columns (beyond Age/Gender) pulled from
# `lead` for the vignette persona. Deliberately excludes LEAD columns that
# are still numeric codes with no documented codebook (economySubjective,
# living1/living2, employment[1-4]) -- see `prompts/vignette.py`'s
# persona_details() docstring for why.
PERSONA_COLUMNS = ["partner", "children", "profession", "employmentType", "education", "economyEarning"]

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
        "vignette_total": PHQ9_PERSONA_ITEMS,
        "persona_total": PHQ9_PERSONA_TOTAL,
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
        "vignette_total": GAD7_PERSONA_ITEMS,
        "persona_total": GAD7_PERSONA_TOTAL,
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
        "vignette_total": PSS10_PERSONA_ITEMS,
        "persona_total": PSS10_PERSONA_TOTAL,
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


def build_prompt(
    variant: str,
    scale_name: str,
    scale: dict,
    item_scores: list[int],
    gender: str | None = None,
    age: int | None = None,
    partner=None,
    children=None,
    profession=None,
    employment_type=None,
    education=None,
    economy_earning=None,
    drop_persona_details: bool = False,
) -> str:
    """Build the prompt for a record under the chosen variant.

    'structured' renders the JSON per-item template (prompts/prompt.py);
    'vignette' renders a first-person persona template (prompts/vignette.py)
    -- goal, persona, item scores as first-person prose, then task; 'vignette_total'
    is the maximal-priming variant (prompts/persona_items.py): same persona
    template, but also names the scale explicitly and states the summed
    total/max score alongside the per-item facts, rather than leaving severity
    implicit in the item facts alone. 'persona_total' (prompts/persona.py) is
    a total-score-only variant: same age/gender persona intro (never the
    extra persona_details() facts, regardless of `drop_persona_details`), and
    it names the scale and states the summed total/max score, but omits the
    per-item breakdown entirely -- the model gets no item-level facts to
    ground its narrative in beyond the single total. Both vignette variants
    need a normalized `gender` category ("female"/"male"/"neutral"), `age`,
    and the raw `partner`/`children`/`profession`/`employmentType`/
    `education`/`economyEarning` values from `lead` (translated into
    persona_details() facts; missing values are dropped, not invented).

    `drop_persona_details=True` keeps age/gender in the persona sentence but
    suppresses the extra persona_details() facts (marital status, children,
    profession, employment, education, income) -- the "age + gender only"
    condition, as opposed to `--drop-persona-intro`'s "no persona at all".
    """
    items = scale["items"]

    if variant == "persona_total":
        total, max_total = total_score(scale_name, items, item_scores)
        return scale["persona_total"].format(
            persona_intro=persona_intro(age, GENDER_NOUNS[gender] if gender else None),
            total_score=round(total, 2),
            max_score=max_total,
            question=QUESTION,
        )

    if variant in ("vignette", "vignette_total"):
        persona = "" if drop_persona_details else persona_details(
            marital_status=partner_phrase(partner),
            children=children_phrase(children),
            profession=clean_freetext(profession),
            employment_type=clean_freetext(employment_type),
            education=education_phrase(education),
            economy_earning=economy_earning_phrase(economy_earning),
        )
        if variant == "vignette":
            return scale["vignette"].format(
                age=age,
                gender=GENDER_NOUNS[gender],
                persona_details=persona,
                item_scores=symptom_narrative_prose(scale_name, items, item_scores),
                question=QUESTION,
            )

        total, max_total = total_score(scale_name, items, item_scores)
        return scale["vignette_total"].format(
            persona_intro=persona_intro(age, GENDER_NOUNS[gender] if gender else None, persona),
            total_score=round(total, 2),
            max_score=max_total,
            item_scores=symptom_narrative_prose(scale_name, items, item_scores),
            question=QUESTION,
        )

    payload = {
        "item_scores": [
            {"item": item, "score": round(score)}
            for item, score in zip(items, item_scores)
        ],
        "question": QUESTION,
    }
    return scale["template"].format(input=json.dumps(payload, indent=2))


# Raw `lead`/`lead_outcomes_agg_*` Gender values -> persona category
# ("Female"/"Male" spelled out; "F"/"M" also accepted in case a different
# source table uses the abbreviated form). Anything else (non-binary, other,
# prefer-not-to-say, ...) falls back to a neutral they/them persona --
# intentional, since there's no single non-binary persona category to map
# them onto.
GENDER_CATEGORY = {"F": "female", "M": "male", "Female": "female", "Male": "male"}
DEFAULT_AGE = 40


def gender_category(raw: str) -> str:
    return GENDER_CATEGORY.get(str(raw).strip(), "neutral")


def generate(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int | None,
    temperature: float,
    max_attempts: int,
) -> tuple[str, str | None]:
    """`max_tokens=None` omits `max_completion_tokens` from the request
    entirely, rather than sending an explicit cap -- so a prompt's own
    word-count instruction (e.g. persona_items.PHQ9's) governs length
    without the model being truncated mid-sentence by a lower,
    independently-set API-side limit."""
    token_kwargs = {} if max_tokens is None else {"max_completion_tokens": max_tokens}
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
                    temperature=temperature,
                    **token_kwargs,
                    reasoning_effort="none"
                )
        return resp.choices[0].message.content or "", None
    except Exception as e:
        return "", repr(e)


def get_source_labels(
    table: str,
    columns: list[str]
) -> pd.DataFrame:
    """Load one row per source record to drive prompt building.

    Two shapes of source table are supported, detected by whether `table`
    itself has a `participant_id` column:

    - Message-level (e.g. lead_en, lead_en_valid): per-item scores come from
      the source table (`a`); demographics live in the raw `lead` table
      (`b`). Inner-joining on message_id filters to the source rows while
      pulling demographics across, so a score split such as lead_en_valid
      (which lacks demographics) can still drive the vignette.
    - Participant-level (e.g. lead_outcomes_agg_valid): item scores and
      demographics are already aggregated onto the same row (mean for
      scores, mode for demographics), so no join is needed -- and none
      should be attempted, since these tables' own `message_id` column (if
      present) is just the meaningless per-participant mean of message ids,
      not a real key back into `lead`.
    """
    engine = create_engine(
        "mysql://ssubrahmanya@/ssubrahmanya?charset=utf8mb4",
        connect_args={"read_default_file": str(Path.home() / ".my.cnf")},
    )

    demographic_cols = ["Age", "Gender"] + PERSONA_COLUMNS
    table_cols = {c["name"] for c in inspect(engine).get_columns(table)}

    if "participant_id" in table_cols:
        id_field = "participant_id"
        select = ", ".join([id_field] + columns + demographic_cols)
        query = f"SELECT {select} FROM {table}"
    else:
        id_field = "message_id"
        select = ", ".join([f"a.{c}" for c in [id_field] + columns] + [f"b.{c}" for c in demographic_cols])
        query = f"SELECT {select} FROM {table} a, lead b WHERE a.{id_field} = b.{id_field}"

    df = pd.read_sql(query, engine)
    df.attrs["id_field"] = id_field
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate responses for LLM benchmarking.")
    parser.add_argument("--scale", choices=sorted(SCALES.keys()), required=True, help="Which scale to prompt with.")
    parser.add_argument("--prompt-variant", choices=["structured", "vignette", "vignette_total", "persona_total"], default="structured", help="Prompt family: 'structured' (prompts/prompt.py), 'vignette' (per-item scores, prompts/vignette.py), 'vignette_total' (summed score + per-item facts, prompts/persona_items.py), or 'persona_total' (summed score only, no per-item breakdown, prompts/persona.py).")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Model names to benchmark (provider-qualified for openrouter, e.g. openai/gpt-4.1-mini).")
    parser.add_argument("--source-table", default="lead_en", help="MySQL table to pull per-item scores from.")
    parser.add_argument("--max-tokens", type=int, default=600, help="Max completion tokens; pass 0 or a negative value to omit the cap entirely and let the model finish naturally (e.g. for prompts with their own relaxed length instruction).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-attempts", type=int, default=6, help="Max retry attempts per request on 429/5xx/timeouts.")
    parser.add_argument("--ntrials", type=int, default=1, help="Number of independent generations per (model, source row), to estimate a within-sample SE and marginalize out generation stochasticity.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and count rows without calling the LLM (smoke-test prompt construction/--ntrials only).")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N source rows (smoke-testing a small number of real LLM calls before a full run).")
    parser.add_argument("--drop-persona-intro", action="store_true", help="For 'vignette'/'vignette_total': omit the '{persona_intro}' sentence entirely (age=None, gender=None), which also drops every persona_details() fact folded into it (marital status, children, profession, employment, education, income). Adds a '_no_persona' suffix to the output filename.")
    parser.add_argument("--drop-persona-details", action="store_true", help="For 'vignette'/'vignette_total': keep age/gender in the persona sentence, but omit the rest of persona_details() (marital status, children, profession, employment, education, income) -- the 'age + gender only' condition. Adds a '_demo_only' suffix to the output filename. Mutually exclusive with --drop-persona-intro.")
    args = parser.parse_args()

    if args.drop_persona_intro and args.drop_persona_details:
        raise SystemExit("--drop-persona-intro already omits age/gender, so --drop-persona-details is redundant with it")

    # setting up necessary logging
    setup_logging()

    # load the API tokens from the .env file
    load_dotenv()
    if not args.dry_run and "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY is not set (add to environment or .env)")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scale = SCALES[args.scale]
    sources = get_source_labels(args.source_table, scale["columns"])
    id_field = sources.attrs["id_field"]
    if args.limit is not None:
        sources = sources.head(args.limit)
    logging.info(f"loaded {len(sources)} rows from {args.source_table} for {args.scale} (id field: {id_field})")

    # The vignette variants personalize each record with its own Age/Gender
    # (pulled from the source table by get_source_labels); structured has none.
    if args.prompt_variant in ("vignette", "vignette_total", "persona_total") and not args.drop_persona_intro:
        missing = int(sources["Gender"].isna().sum())
        if missing:
            logging.warning("%d rows missing demographics; using neutral persona / age %d", missing, DEFAULT_AGE)
        cats = sources["Gender"].map(gender_category)
        logging.info("persona categories: %s", cats.value_counts().to_dict())
    elif args.drop_persona_intro:
        logging.info("--drop-persona-intro set: every row gets gender=None, age=None (no persona sentence)")

    client = None if args.dry_run else OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )

    rows = []
    for model in args.models:

        logging.info(f"generating responses for model {model}...")
        for idx, record in tqdm(sources.iterrows(), total=len(sources), desc=f"{args.scale}/{args.prompt_variant}"):

            # Kept as raw (possibly float, for participant-level mean-aggregated
            # sources) rather than rounded here -- total_score() needs the raw
            # values to avoid compounding rounding error across items; the
            # per-item narrative (vignette variant) rounds where it's used.
            item_scores = [record[col] for col in scale["columns"]]

            if args.prompt_variant in ("vignette", "vignette_total", "persona_total"):
                if args.drop_persona_intro:
                    gender, age = None, None
                else:
                    gender = gender_category(record["Gender"])
                    age = round(record["Age"]) if pd.notna(record["Age"]) else DEFAULT_AGE
                prompt = build_prompt(
                    args.prompt_variant, args.scale, scale, item_scores, gender, age,
                    record["partner"], record["children"], record["profession"],
                    record["employmentType"], record["education"], record["economyEarning"],
                    drop_persona_details=args.drop_persona_details,
                )
            else:
                gender, age = None, None
                prompt = build_prompt(
                    args.prompt_variant, args.scale, scale, item_scores, gender, age,
                )

            for trial in range(args.ntrials):
                if args.dry_run:
                    text, err = "", None
                else:
                    text, err = generate(
                        client,
                        model,
                        prompt,
                        args.max_tokens if args.max_tokens > 0 else None,
                        args.temperature,
                        args.max_attempts,
                    )

                rows.append(
                    {
                        id_field: record[id_field],
                        "model": model,
                        "scale": args.scale,
                        "variant": args.prompt_variant,
                        "trial": trial,
                        "gender": gender,
                        "age": age,
                        "item_scores": json.dumps(item_scores),
                        "prompt": prompt,
                        "text": text,
                        "error": err,
                    }
                )

    suffix = "" if args.prompt_variant == "structured" else f"_{args.prompt_variant}"
    if args.drop_persona_intro:
        suffix += "_no_persona"
    if args.drop_persona_details:
        suffix += "_demo_only"
    output_path = args.output_dir / f"generated_responses_{args.scale}{suffix}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logging.info("wrote %s", output_path)