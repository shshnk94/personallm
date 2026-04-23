"""Prompt templates for PHQ-9 calibration generation."""

SYSTEM_PROMPT = (
    "You are helping build a research dataset that calibrates language models against the PHQ-9 "
    "depression screening instrument. Each PHQ-9 item is scored 0-3 (0=not at all, 1=several days, "
    "2=more than half the days, 3=nearly every day) across 9 symptoms: (1) anhedonia, (2) depressed mood, "
    "(3) sleep problems, (4) fatigue, (5) appetite change, (6) worthlessness/guilt, (7) concentration, "
    "(8) psychomotor change, (9) suicidal ideation. Total score ranges 0-27. Severity bands: "
    "0-4 none/minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-27 severe."
)

USER_TEMPLATE = (
    "Write a short first-person journal entry (120-180 words) from an anonymous person whose current "
    "PHQ-9 total score is {score}/27. The narrative should implicitly reflect symptom severity "
    "consistent with that total (spread across the 9 symptom domains in a realistic way). "
    "Do NOT mention PHQ-9, scores, numbers, or clinical terminology. Do not add disclaimers. "
    "Return only the journal entry text."
)
