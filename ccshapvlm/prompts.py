"""Model-independent prompt text: task question templates and the answer phrases.

The dialect-specific scaffolding (chat message structure, instruction tokens, the
first-person vs second-person chain-of-thought wording) lives on the model adapters in
``ccshapvlm.models``; here we keep only the pieces every adapter shares verbatim.
"""

from __future__ import annotations

PHRASE_ANSWER_MULTIPLE_CHOICE = "The best answer is:"
PHRASE_ANSWER_OPEN_ENDED = "The best short answer is:"


def format_example_valse(caption: str) -> str:
    return (f'Here is a tentative caption for the image: "{caption}". Does the caption '
            "accurately describe the image or is there something wrong with it? Choose one "
            "of the following answers: (A): The caption is correct; (B): The caption is "
            "incorrect.")


def format_example_valse_pairwise(caption: str, foil: str) -> str:
    return (f'Which caption is a correct description of the image? Is it (A): "{caption}" '
            f'or is it (B): "{foil}"?')


def format_example_vqa_gqa(question: str) -> str:
    return question
