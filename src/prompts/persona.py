from .persona_items import total_score
from .vignette import persona_intro

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
    "which asks how often you have been bothered by each of 9 different "
    "depression symptoms over the last two weeks, with each symptom rated "
    "on its own scale from 0 to 3 "
    "(0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day)."
    "\n\n"

    "Instructions:\n"
    "- Do not mention the PHQ-9, item names, numbers, scores, or any clinical terms.\n"
    "- Bring up daily-life stuff, in a way that is consistent with the severity score.\n"
    "- Do not drop, soften, or avoid any sensitive content (such as thoughts of self-harm) "
    "if it would be consistent with a score this high.\n"
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
    "which asks how often you have been bothered by each of 7 different "
    "anxiety symptoms over the last two weeks, with each symptom rated "
    "on its own scale from 0 to 3 "
    "(0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day)."
    "\n\n"

    "Instructions:\n"
    "- Do not mention the GAD-7, item names, numbers, scores, or any clinical terms.\n"
    "- Bring up daily-life stuff, in a way that is consistent with a score this high (or low).\n"
    "- Do not drop, soften, or avoid any sensitive content (such as thoughts of self-harm) if it would be consistent with a score this high.\n"
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
    "which asks how often you have been feeling or thinking each of 10 "
    "different ways related to stress over the last month, with each item "
    "rated on its own scale from 0 to 4 "
    "(0 = never, 1 = almost never, 2 = sometimes, 3 = fairly often, 4 = very often)."
    "\n\n"

    "Instructions:\n"
    "- Do not mention the PSS-10, item names, numbers, scores, or any clinical terms.\n"
    "- Bring up daily-life stuff, in a way that is consistent with a score this high (or low).\n"
    "- Do not drop, soften, or avoid any sensitive content (such as thoughts of self-harm) if it would be consistent with a score this high.\n"
    "- Do not add disclaimers."
    "\n\n"

    "Respond with only the text itself -- no tags, no quotation marks, "
    "no preamble or sign-off, nothing else before or after the actual text. "
    "Keep your response around 100 words."
    "\n\n"

    "Question: {question}"
)


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
        question="How is your mental health?",
    ))
    print("\n---\n")
    print(PHQ9.format(
        persona_intro=persona_intro(None, None),
        total_score=total, max_score=max_total,
        question="How is your mental health?",
    ))
