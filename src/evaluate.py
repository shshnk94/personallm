"""PHQ-9 calibration — stage 2: score generations with a depression classifier.

Reads <output-dir>/generations.csv, writes scored.csv, aggregate.csv,
summary.csv, calibration.png. Pass --fresh to re-score when scored.csv exists.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PHQ9_MIN, PHQ9_MAX = 0, 27

SCORER_MODEL = "rafalposwiata/deproberta-large-depression"
SEVERITY_TO_PHQ9 = {"not depression": 2.0, "moderate": 12.0, "severe": 22.0}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def load_scorer(device: str):
    tokenizer = AutoTokenizer.from_pretrained(SCORER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SCORER_MODEL).to(device).eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    midpoints = torch.tensor(
        [SEVERITY_TO_PHQ9[id2label[i]] for i in range(len(id2label))],
        device=device,
        dtype=torch.float32,
    )
    return tokenizer, model, midpoints


@torch.inference_mode()
def score_texts(texts: list[str], tokenizer, model, midpoints: torch.Tensor, batch_size: int) -> np.ndarray:
    device = midpoints.device
    out = np.zeros(len(texts), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        out[i : i + len(batch)] = (probs * midpoints).sum(dim=-1).cpu().numpy()
    return out


def summarize(df: pd.DataFrame) -> dict:
    t, p = df["target_score"].to_numpy(), df["predicted_score"].to_numpy()
    return {
        "n": len(df),
        "pearson_r": float(np.corrcoef(t, p)[0, 1]) if len(df) > 1 else float("nan"),
        "mae": float(np.mean(np.abs(t - p))),
        "bias": float(np.mean(p - t)),
    }


def plot_calibration(agg: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for model, sub in agg.groupby("model"):
        ax.errorbar(sub["target_score"], sub["mean"], yerr=sub["std"], marker="o", capsize=2, label=model, alpha=0.85)
    ax.plot([PHQ9_MIN, PHQ9_MAX], [PHQ9_MIN, PHQ9_MAX], "k--", alpha=0.4, label="y = x")
    ax.set_xlabel("Target PHQ-9 score")
    ax.set_ylabel("Predicted PHQ-9 score (mean ± std)")
    ax.set_title("LLM calibration against PHQ-9")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PHQ-9 benchmark — evaluation stage")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Directory containing generations.csv; outputs written here too.")
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--fresh", action="store_true", help="Ignore cached scored.csv and re-score.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    setup_logging()

    gen_path = args.output_dir / "generations.csv"
    scored_path = args.output_dir / "scored.csv"
    agg_path = args.output_dir / "aggregate.csv"
    summary_path = args.output_dir / "summary.csv"
    plot_path = args.output_dir / "calibration.png"

    if not gen_path.exists():
        raise SystemExit(f"missing {gen_path}; run generate.py first")

    if args.fresh:
        scored_path.unlink(missing_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("device=%s", device)

    if scored_path.exists():
        valid = pd.read_csv(scored_path)
        logging.info("loaded cached scores: %d rows", len(valid))
    else:
        gens = pd.read_csv(gen_path)
        failed = int(gens["error"].notna().sum())
        if failed:
            logging.warning("%d failed generations dropped before scoring", failed)
        valid = gens[gens["error"].isna() & gens["text"].str.len().gt(0)].reset_index(drop=True).copy()
        logging.info("scoring %d texts with %s", len(valid), SCORER_MODEL)
        tokenizer, scorer, midpoints = load_scorer(device)
        valid["predicted_score"] = score_texts(valid["text"].tolist(), tokenizer, scorer, midpoints, args.score_batch_size)
        valid.to_csv(scored_path, index=False)
        logging.info("wrote %s", scored_path)

    agg = (
        valid.groupby(["model", "target_score"])["predicted_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.to_csv(agg_path, index=False)

    summary = pd.DataFrame({m: summarize(sub) for m, sub in valid.groupby("model")}).T
    summary.to_csv(summary_path)
    logging.info("per-model summary:\n%s", summary.to_string())

    plot_calibration(agg, plot_path)
    logging.info("wrote %s", plot_path)


if __name__ == "__main__":
    main()
