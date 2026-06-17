"""Vignette-framed prompts for the three scales, with deterministic phrasing.

The templates present an already rendered third-person clinical case summary and
ask the model to role-play the patient in the first person. Every slot --
`name`, `age`, `gender`, `pronoun`, `verb`, `severity_label`, `score`,
`symptom_narrative`, `functioning` -- is filled in by the caller; the model only
writes the first-person paragraph. The persona attributes are keyed by a
normalized gender category ("female" / "male" / "neutral") via `NAMES`,
`PRONOUNS`, `VERBS`, and `GENDER_NOUNS`; "neutral" yields a they/them persona.

The `symptom_narrative` is built by `symptom_narrative()` from a fixed
score->intensity dictionary (1 -> "mild", 2 -> "moderate", 3 -> "severe"; items
scored 0 are dropped) rather than by the LLM, so the symptom list is
deterministic and reproducible. For PSS-10 the four reverse-scored items are
inverted first (a low raw score means more stress) before mapping.

The total score and severity band are surfaced as a conditioning signal. They
are not used as a training signal, so they condition the generation without
leaking into the downstream regressor. `functioning` is an optional impairment
sentence that, when present, must carry its own leading space (pass "" to omit
it cleanly for low-severity bands).
"""

PHQ9_VIGNETTE = (
    "Your task is to write a response to a question, as if you were the patient "
    "described in the clinical case summary below. The summary describes a "
    "fictional persona for a research study on how people describe their mental "
    "health."
    "\n\n"

    "Case summary:\n"
    "'{name} is a {age}-year-old {gender} who presents with {severity_label} depression "
    "(PHQ-9: {score}). {pronoun} {verb} {symptom_narrative}.{functioning}'\n\n"
    "Now write a single first-person paragraph as this patient, answering the "
    "question below. Write in the patient's own voice and stay faithful to the "
    "case summary. Do not mention the PHQ-9, the numeric score, the severity "
    "category, item names, numbers, or any clinical terms. Do not add disclaimers."
    "\n\n"

    "Format your output as a JSON with two fields: 'reason' briefly noting which "
    "parts of the case summary you expressed and how, and 'text-response' "
    "containing the final first-person paragraph. Elements inside angle brackets "
    "are placeholders for the actual values:\n"
    "{{\n"
    "    \"reason\": \"<short explanation of how the case summary was expressed>\",\n"
    "    \"text-response\": \"<single first-person paragraph>\"\n"
    "}}"
    "\n\n"

    "Question: {question}"
)


GAD7_VIGNETTE = (
    "Your task is to write a response to a question, as if you were the patient "
    "described in the clinical case summary below. The summary describes a "
    "fictional persona for a research study on how people describe their mental "
    "health."
    "\n\n"

    "Case summary:\n"
    "'{name} is a {age}-year-old {gender} who presents with {severity_label} anxiety "
    "(GAD-7: {score}). {pronoun} {verb} {symptom_narrative}.{functioning}'\n\n"
    "Now write a single first-person paragraph as this patient, answering the "
    "question below. Write in the patient's own voice and stay faithful to the "
    "case summary. Do not mention the GAD-7, the numeric score, the severity "
    "category, item names, numbers, or any clinical terms. Do not add disclaimers."
    "\n\n"

    "Format your output as a JSON with two fields: 'reason' briefly noting which "
    "parts of the case summary you expressed and how, and 'text-response' "
    "containing the final first-person paragraph. Elements inside angle brackets "
    "are placeholders for the actual values:\n"
    "{{\n"
    "    \"reason\": \"<short explanation of how the case summary was expressed>\",\n"
    "    \"text-response\": \"<single first-person paragraph>\"\n"
    "}}"
    "\n\n"

    "Question: {question}"
)


PSS10_VIGNETTE = (
    "Your task is to write a response to a question, as if you were the patient "
    "described in the clinical case summary below. The summary describes a "
    "fictional persona for a research study on how people describe their mental "
    "health."
    "\n\n"

    "Case summary:\n"
    "'{name} is a {age}-year-old {gender} who presents with {severity_label} perceived "
    "stress (PSS-10: {score}). {pronoun} {verb} {symptom_narrative}.{functioning}'\n\n"
    "Now write a single first-person paragraph as this patient, answering the "
    "question below. Write in the patient's own voice and stay faithful to the "
    "case summary. Do not mention the PSS-10, the numeric score, the stress "
    "category, item names, numbers, or any clinical terms. Do not add disclaimers."
    "\n\n"

    "Format your output as a JSON with two fields: 'reason' briefly noting which "
    "parts of the case summary you expressed and how, and 'text-response' "
    "containing the final first-person paragraph. Elements inside angle brackets "
    "are placeholders for the actual values:\n"
    "{{\n"
    "    \"reason\": \"<short explanation of how the case summary was expressed>\",\n"
    "    \"text-response\": \"<single first-person paragraph>\"\n"
    "}}"
    "\n\n"

    "Question: {question}"
)


# Persona attributes per normalized gender category. `pronoun` is
# sentence-initial and `verb` agrees with it ("She reports ...", "He reports
# ...", "They report ..."); `gender` is the noun used in the case summary
# ("a 40-year-old female / male / person who presents ...").
NAMES = {"female": "Alice", "male": "Bob", "neutral": "Sam"}
PRONOUNS = {"female": "She", "male": "He", "neutral": "They"}
VERBS = {"female": "reports", "male": "reports", "neutral": "report"}
GENDER_NOUNS = {"female": "female", "male": "male", "neutral": "person"}


# --- Deterministic symptom phrasing -----------------------------------------
# score -> intensity adjective. Items scored 0 are dropped (no entry for 0).
INTENSITY = {
    "phq9": {1: "mild", 2: "moderate", 3: "severe"},
    "gad7": {1: "mild", 2: "moderate", 3: "severe"},
    # PSS-10 items run 0-4.
    "pss10": {1: "mild", 2: "moderate", 3: "marked", 4: "severe"},
}

# item name (as in generate.py SCALES["<scale>"]["items"]) -> symptom phrase.
PHRASES = {
    "phq9": {
        "Anhedonia": "loss of interest in things",
        "Depressed Mood": "depressed mood",
        "Insomnia or Hypersomnia": "sleep disturbance",
        "Fatigue": "fatigue",
        "Poor Appetite or Overeating": "appetite disturbance",
        "Worthlessness or Guilt": "feelings of worthlessness or guilt",
        "Difficulty Concentrating": "difficulty concentrating",
        "Psychomotor Agitation or Retardation": "psychomotor agitation or slowing",
        "Suicidal Ideation": "suicidal ideation",
    },
    "gad7": {
        "Nervousness": "nervousness",
        "Uncontrollable Worry": "uncontrollable worry",
        "Excessive Worry": "excessive worry",
        "Trouble Relaxing": "difficulty relaxing",
        "Restlessness": "restlessness",
        "Irritability": "irritability",
        "Apprehension": "apprehension that something awful might happen",
    },
    "pss10": {
        "Upset by Unexpected Events": "feeling upset by unexpected events",
        "Lack of Control": "a lack of control over important things",
        "Nervous and Stressed": "feeling nervous and stressed",
        "Confidence in Coping (reverse)": "low confidence in coping",
        "Things Going Your Way (reverse)": "a sense that things are not going well",
        "Inability to Cope": "an inability to cope with demands",
        "Control of Irritations (reverse)": "difficulty controlling irritations",
        "On Top of Things (reverse)": "feeling unable to stay on top of things",
        "Anger at Things Outside Control": "anger over things beyond control",
        "Overwhelming Difficulties": "a sense of overwhelming difficulties",
    },
}

# Reverse-scored PSS-10 items: a LOWER raw score means MORE stress, so invert
# the raw score (max - raw) before looking up its intensity.
REVERSE = {
    "phq9": set(),
    "gad7": set(),
    "pss10": {
        "Confidence in Coping (reverse)",
        "Things Going Your Way (reverse)",
        "Control of Irritations (reverse)",
        "On Top of Things (reverse)",
    },
}

# Max raw score per scale, used to invert reverse-scored items.
MAX_SCORE = {"phq9": 3, "gad7": 3, "pss10": 4}


def symptom_narrative(scale: str, items: list[str], scores: list[int]) -> str:
    """Compose a deterministic, comma-joined symptom list from item scores.

    Each reportable item becomes "<intensity> <phrase>" (e.g. "severe sleep
    disturbance"); items with no stress signal are dropped. Reverse-scored
    PSS-10 items are inverted first. Returns "" if nothing is reportable.
    """
    intensity, phrases = INTENSITY[scale], PHRASES[scale]
    reverse, max_score = REVERSE[scale], MAX_SCORE[scale]
    parts = []
    for item, raw in zip(items, scores):
        level = (max_score - int(raw)) if item in reverse else int(raw)
        if level <= 0:
            continue
        parts.append(f"{intensity[level]} {phrases[item]}")
    return ", ".join(parts)


if __name__ == "__main__":
    phq9_items = list(PHRASES["phq9"])
    demo = [2, 3, 3, 2, 1, 2, 1, 0, 2]
    for cat in ("female", "male", "neutral"):
        print(PHQ9_VIGNETTE.format(
            name=NAMES[cat], age=45, gender=GENDER_NOUNS[cat],
            pronoun=PRONOUNS[cat], verb=VERBS[cat],
            severity_label="moderately severe", score=16,
            symptom_narrative=symptom_narrative("phq9", phq9_items, demo),
            functioning=" Functioning is significantly impaired.",
            question="How is your mental health?",
        ))
        print("\n---\n")