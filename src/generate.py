"""PHQ-9 calibration — stage 1: generate journal entries per (model, target score).

Writes <output-dir>/generations.csv. Resume-safe: re-running skips if the
CSV exists unless --fresh is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import asyncio
import logging
import os
import sys


import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tqdm.asyncio import tqdm_asyncio

from .prompt import SYSTEM_PROMPT, USER_TEMPLATE

PHQ9_MIN, PHQ9_MAX = 0, 27

MODELS = [
    "google/gemini-2.5-flash-lite",
    # "openai/gpt-4.1-nano",
    # "anthropic/claude-haiku-4-5",
    # "meta-llama/llama-3.3-70b-instruct",
]

RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def build_messages(target_score: int) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(score=target_score)},
    ]


async def generate(
    client: AsyncOpenAI,
    model: str,
    target: int,
    sem: asyncio.Semaphore,
    max_tokens: int,
    temperature: float,
    max_attempts: int,
) -> dict:
    async with sem:
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(RETRYABLE),
                wait=wait_exponential_jitter(initial=1, max=30),
                stop=stop_after_attempt(max_attempts),
                before_sleep=before_sleep_log(logging.getLogger(), logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=build_messages(int(target)),
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
            text = resp.choices[0].message.content or ""
            err = None
        except Exception as e:
            text, err = "", repr(e)
        return {"model": model, "target_score": int(target), "text": text, "error": err}


async def generate_prompts(
    client: AsyncOpenAI,
    models: list[str],
    targets: np.ndarray,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    max_attempts: int,
) -> pd.DataFrame:

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        generate(
            client,
            model,
            int(target),
            semaphore,
            max_tokens,
            temperature,
            max_attempts,
        ) for model in models for target in targets
    ]
    rows = await tqdm_asyncio.gather(*tasks, desc="generating")
    rows = pd.DataFrame(rows)

    return rows

def generate_score_sample(samples_per_score: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    targets = np.repeat(np.arange(PHQ9_MIN, PHQ9_MAX + 1), samples_per_score)
    rng.shuffle(targets)
    return targets

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Generate journal entries for LLM benchmarking.")

    # Add arguments
    parser.add_argument("--models", nargs="+", default=MODELS, help="OpenRouter model slugs to benchmark.")
    parser.add_argument("--samples-per-score", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=8, help="Max in-flight requests per model group.")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=6, help="Max retry attempts per request on 429/5xx/timeouts.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    
    args = parser.parse_args()
    return args


def main() -> None:

    args = parse_arguments()

    # pre-flight checks and setup
    setup_logging()

    load_dotenv()
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY is not set (add to environment or .env)")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Make it reusable for other questionnaires
    targets = generate_score_sample(args.samples_per_score, args.seed)
    logging.info(f"generating {len(targets)} prompts per model")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )
    gens = asyncio.run(
        generate_prompts(
            client,
            args.models,
            targets,
            args.concurrency,
            args.max_tokens,
            args.temperature,
            args.max_attempts,
        )
    )
    gens.to_csv(output_dir / "generated_prompts.csv", index=False)
    logging.info("wrote %s", output_dir)


if __name__ == "__main__":
    main()
