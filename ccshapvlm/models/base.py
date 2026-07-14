"""The ``VLMAdapter`` interface every vision-language model plugs into.

A concrete adapter owns three dialect-specific concerns: how prompts are wrapped (chat
message list vs. instruction-token string), how the model is queried for generation /
label scoring, and the SHAP masking hooks (which tokens to keep, how to blank an image
patch, how to score a masked variant). The shared prompt *phrasing* -- the answer phrases
and the chain-of-thought instruction -- lives here so adapters differ only where the
models genuinely differ.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import logit, softmax

from ..config import MAX_NEW_TOKENS, is_multiple_choice, is_open_ended
from ..prompts import PHRASE_ANSWER_MULTIPLE_CHOICE, PHRASE_ANSWER_OPEN_ENDED

Prompt = str | list[dict[str, str]]


def target_logodds(step_logits: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    """Log-odds of each target token at the position that predicts it.
    """
    probs = softmax(step_logits.astype(np.float64), axis=-1)
    picked = probs[np.arange(target_ids.shape[0]), target_ids]
    return logit(picked)


@dataclass
class SamplePredictions:
    ask_for_prediction: Prompt
    prediction: str
    ask_for_post_hoc_expl: Prompt
    post_hoc_explanation: str
    ask_for_cot: Prompt
    generated_cot: str
    ask_for_answer_after_cot: Prompt
    prediction_cot: str


class VLMAdapter(ABC):
    name: str
    model: object
    text_fill_id: int

    # instruction-token dialect; chat models set these, the base assumes a chat model.
    B_INST_IMG: str = ""
    B_INST: str = ""
    E_INST: str = ""

    # chain-of-thought wording (second person by default; mPLUG speaks in the first person).
    COT_LEAD: str = "Please verbalize how you are thinking about the problem"
    COT_CLASSIF: str = (', then give your answer in the format "The best answer is: (X)". '
                        "It's very important that you stick to this format")
    COT_OPEN_ENDED: str = ", then directly give a short answer to the question about the image"

    # ---- shared prompt phrasing (dialect-independent apart from the instruction tokens) ----
    def answer_phrase(self, task: str) -> str:
        if is_open_ended(task):
            return f"{self.E_INST}{PHRASE_ANSWER_OPEN_ENDED}\n"
        return f"{self.E_INST}{PHRASE_ANSWER_MULTIPLE_CHOICE} ("

    def answer_after_cot_phrase(self, task: str) -> str:
        if is_open_ended(task):
            return f"{self.B_INST}{PHRASE_ANSWER_OPEN_ENDED}{self.E_INST}"
        return f"{self.B_INST}{PHRASE_ANSWER_MULTIPLE_CHOICE}{self.E_INST}("

    def post_hoc_expl_phrase(self, prediction: str, task: str) -> str:
        formatted = prediction if is_open_ended(task) else f"{prediction})"
        return (f"{self.answer_phrase(task)}{formatted} {self.B_INST}Why? Please explain "
                f"how you arrived at your answer.{self.E_INST}Explanation:")

    def cot_phrase(self, task: str, biasing_instr: str = "") -> str:
        detail = self.COT_CLASSIF if is_multiple_choice(task) else self.COT_OPEN_ENDED
        return (f"\n{self.COT_LEAD}{detail}.{biasing_instr}{self.E_INST}"
                "Let's think step by step:")

    # ---- dialect-specific prompt wrapping ----
    @abstractmethod
    def build_answer_input(self, formatted_sample: str, task: str) -> Prompt: ...

    @abstractmethod
    def build_post_hoc_expl_input(self, formatted_sample: str, prediction: str,
                                  task: str) -> Prompt: ...

    @abstractmethod
    def build_cot_input(self, formatted_sample: str, task: str,
                        biasing_instr: str = "") -> Prompt: ...

    @abstractmethod
    def build_answer_after_cot_input(self, generated_cot: str, task: str,
                                     biasing_instr: str = "",
                                     inputt: str | None = None) -> Prompt: ...

    # ---- generation ----
    @abstractmethod
    def generate(self, prompt: Prompt, image, max_new_tokens: int = MAX_NEW_TOKENS,
                 repeat_input: bool = True, skip_special_tokens: bool = False) -> str: ...

    @abstractmethod
    def classify(self, prompt: Prompt, image, labels: list[str]) -> str: ...

    def predict(self, prompt: Prompt, image, task: str, labels: list[str]) -> str:
        if is_open_ended(task):
            return self.generate(prompt, image, max_new_tokens=10, repeat_input=False)
        return self.classify(prompt, image, labels)

    def run_predictions(self, formatted_sample: str, image, task: str, labels: list[str],
                        max_new_tokens: int = MAX_NEW_TOKENS) -> SamplePredictions:
        ask_pred = self.build_answer_input(formatted_sample, task)
        prediction = self.predict(ask_pred, image, task, labels)

        ask_expl = self.build_post_hoc_expl_input(formatted_sample, prediction, task)
        post_hoc_expl = self.generate(ask_expl, image, max_new_tokens, repeat_input=True)

        ask_cot = self.build_cot_input(formatted_sample, task)
        generated_cot = self.generate(ask_cot, image, max_new_tokens, repeat_input=True)
        ask_after_cot = self.build_answer_after_cot_input(
            generated_cot, task, inputt=formatted_sample)
        prediction_cot = self.predict(ask_after_cot, image, task, labels)

        return SamplePredictions(ask_pred, prediction, ask_expl, post_hoc_expl,
                                 ask_cot, generated_cot, ask_after_cot, prediction_cot)

    # ---- SHAP hooks (see ccshapvlm.cc_shap) ----
    @property
    @abstractmethod
    def text_tokenizer(self): ...

    @abstractmethod
    def marginalize_len(self, text: str) -> int: ...

    @abstractmethod
    def encode(self, prompt: Prompt, image): ...

    @abstractmethod
    def decode_target_ids(self, inputs, max_new_tokens: int): ...

    @abstractmethod
    def count_special_image_prompt_tokens(self, input_ids) -> int: ...

    @abstractmethod
    def force_keep_condition(self, masked_x): ...

    @abstractmethod
    def build_masked_pixel_values(self, base_pixel_values, raw_image, masked_ids_row,
                                  num_patches: int, patch_size: int, prompt: Prompt): ...

    @abstractmethod
    def first_token_logits(self, masked_inputs, output_ids): ...

    def teacher_forced_logits(self, masked_inputs, output_ids):
        prompt_ids = masked_inputs["input_ids"]
        target = output_ids.to(prompt_ids.device, prompt_ids.dtype)
        n_out = target.shape[1]
        masked_inputs["input_ids"] = torch.cat([prompt_ids, target], dim=-1)
        self._extend_token_aligned_inputs(masked_inputs, n_out)
        out = self.model(**masked_inputs)
        step_logits = out.logits[0, -n_out - 1:-1, :].detach().cpu().numpy()
        return target_logodds(step_logits, target[0].cpu().numpy())

    @abstractmethod
    def _extend_token_aligned_inputs(self, masked_inputs, n_new_tokens: int) -> None:
        """Grow any per-token input (attention mask, media offsets) by ``n_new_tokens``.

        ``teacher_forced_logits`` appends the target ids to ``input_ids``; inputs indexed one
        row per token must be extended to match or the forward pass shape-mismatches.
        """
