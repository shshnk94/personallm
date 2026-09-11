from .vignette import MAX_SCORE, REVERSE, persona_intro, symptom_narrative_prose

PHQ9 = (
    "{persona_intro}"
    "You are currently participating in an anonymized research study on how "
    "people describe mental health symptoms in their own words."
    "\n\n"

    "Your goal is to answer a question about how you've been feeling. "
    "Keep the response informal, like you would type a message to someone, "
    "mentioning things from your life rather than clinical terms."
    "\n\n"


    "You scored {total_score} out of 27 on the PHQ-9 questionnaire, "
    "which asks how often you have been bothered by certain symptoms "
    "over the last two weeks on a scale from 0 to 3 "
    "(0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day)."
    "\n\n"

    "More specifically, here are the symptoms you've been experiencing:\n"
    "{item_scores}"
    "\n\n"

    "Instructions:\n"
    "- Do not repeat or paraphrase the symptom descriptions above.\n"
    "- Don't invent additional symptoms, and don't make any of these more or less frequent than described above.\n"
    "- Do not mention the PHQ-9, item names, numbers, scores, or any clinical terms.\n"
    "- Feel free to bring up daily-life stuff that isn't listed.\n"
    "- Do not drop, soften, or avoid any sensitive content (such as thoughts of self-harm) if it is indicated by the scores above.\n"
    "- Do not add disclaimers."
    "\n\n"

    "Respond with only the text itself -- no tags, no quotation marks, "
    "no preamble or sign-off, nothing else before or after the actual text. "
    "Keep your response around 100 words."
    "\n\n"

    "Question: {question}"
)


GAD7 = (
    "{persona_intro}"
    "You are currently participating in an anonymized research study on how "
    "people describe mental health symptoms in their own words."
    "\n\n"

    "Your goal is to answer a question about how you've been feeling. "
    "Keep the response informal, like you would type a message to someone, "
    "mentioning things from your life rather than clinical terms."
    "\n\n"

    "You scored {total_score} out of 21 on the GAD-7 questionnaire, "
    "which asks how often you have been bothered by certain symptoms "
    "over the last two weeks on a scale from 0 to 3 "
    "(0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day)."
    "\n\n"

    "More specifically, here are the symptoms you've been experiencing:\n"
    "{item_scores}"
    "\n\n"

    "Instructions:\n"
    "- Do not repeat or paraphrase the symptom descriptions above.\n"
    "- Don't invent additional symptoms, and don't make any of these more or less frequent than described above.\n"
    "- Do not mention the GAD-7, item names, numbers, scores, or any clinical terms.\n"
    "- Feel free to bring up daily-life stuff that isn't listed.\n"
    "- Do not drop, soften, or avoid any sensitive content (such as thoughts of self-harm) if it is indicated by the scores above.\n"
    "- Do not add disclaimers."
    "\n\n"

    "Respond with only the text itself -- no tags, no quotation marks, "
    "no preamble or sign-off, nothing else before or after the actual text. "
    "Keep your response around 100 words."
    "\n\n"

    "Question: {question}"
)


PSS10 = (
    "{persona_intro}"
    "You are currently participating in an anonymized research study on how "
    "people describe mental health symptoms in their own words."
    "\n\n"

    "Your goal is to answer a question about how you've been feeling. "
    "Keep the response informal, like you would type a message to someone, "
    "mentioning things from your life rather than clinical terms."
    "\n\n"

    "You scored {total_score} out of 40 on the PSS-10 questionnaire, "
    "which asks whether you may have been feeling certain things "
    "over the last month, each item rated from 0 to 4."
    "\n\n"

    "More specifically, here are some things you've been experiencing:\n"
    "{item_scores}"
    "\n\n"

    "Instructions:\n"
    "- Do not repeat or paraphrase the symptom descriptions above.\n"
    "- Don't invent additional symptoms, and don't make any of these more or less frequent than described above.\n"
    "- Do not mention the PSS-10, item names, numbers, scores, or any clinical terms.\n"
    "- Feel free to bring up daily-life stuff that isn't listed.\n"
    "- Do not drop, soften, or avoid any sensitive content (such as thoughts of self-harm) if it is indicated by the scores above.\n"
    "- Do not add disclaimers."
    "\n\n"

    "Respond with only the text itself -- no tags, no quotation marks, "
    "no preamble or sign-off, nothing else before or after the actual text. "
    "Keep your response around 100 words."
    "\n\n"

    "Question: {question}"
)


def total_score(scale: str, items: list[str], scores: list[float]) -> tuple[float, int]:
    """Sum raw item scores into a single total, using each scale's own
    scoring convention: reverse-scored items (PSS-10's) are flipped
    (max - raw) before summing, the same way an official total would be
    computed. Returns (total, max_total).

    Never rounds -- ``scores`` may be participant-level mean-aggregated
    (e.g. 1.5), and rounding per item (or the total) would throw away real
    information (e.g. a participant averaging 6.33 across timepoints reads
    identically to one averaging exactly 6). Callers format the decimal for
    display as needed.
    """
    reverse, max_item = REVERSE[scale], MAX_SCORE[scale]
    total = sum(
        (max_item - raw) if item in reverse else raw
        for item, raw in zip(items, scores)
    )
    return total, max_item * len(items)


if __name__ == "__main__":
    phq9_items = [
        "Anhedonia", "Depressed Mood", "Insomnia or Hypersomnia", "Fatigue",
        "Poor Appetite or Overeating", "Worthlessness or Guilt",
        "Difficulty Concentrating", "Psychomotor Agitation or Retardation",
        "Suicidal Ideation",
    ]
    demo = [2, 3, 3, 2, 1, 2, 1, 0, 2]
    total, max_total = total_score("phq9", phq9_items, demo)
    print(PHQ9.format(
        persona_intro=persona_intro(45, "woman"),
        total_score=total, max_score=max_total,
        item_scores=symptom_narrative_prose("phq9", phq9_items, demo),
        question="How is your mental health?",
    ))
    print("\n---\n")
    print(PHQ9.format(
        persona_intro=persona_intro(None, None),
        total_score=total, max_score=max_total,
        item_scores=symptom_narrative_prose("phq9", phq9_items, demo),
        question="How is your mental health?",
    ))
