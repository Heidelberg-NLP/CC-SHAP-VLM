"""The Llama-2 chat helper used by the counterfactual/CoT consistency tests.

The helper edits or paraphrases the model's chain of thought (Turpin, Lanham) and is loaded
only when those tests run; the CC-SHAP measures do not need it.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MAX_NEW_TOKENS, MODELS

_B_INST, _E_INST = "[INST] ", " [/INST]"
_B_SYS, _E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
_SYSTEM = (f"{_B_SYS}You are a helpful chat assistant and will answer the user's "
           f"questions carefully.{_E_SYS}")


class HelperModel:
    def __init__(self, name: str, model, tokenizer):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, name: str) -> "HelperModel":
        with torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(
                MODELS[name], torch_dtype=torch.float16, device_map="auto", token=True)
        tokenizer = AutoTokenizer.from_pretrained(
            MODELS[name], use_fast=False, padding_side="left")
        return cls(name, model, tokenizer)

    def generate(self, text: str, max_new_tokens: int = MAX_NEW_TOKENS,
                 repeat_input: bool = True) -> str:
        input_ids = self.tokenizer([text], return_tensors="pt").input_ids.cuda()
        generated_ids = self.model.generate(
            input_ids, max_new_tokens=max_new_tokens, min_new_tokens=1)
        if not repeat_input:
            generated_ids = generated_ids[:, input_ids.shape[1]:]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def chat(self, user_message: str, assistant_prefix: str, max_new_tokens: int) -> str:
        prompt = f"{_SYSTEM}{_B_INST}{user_message}{_E_INST} {assistant_prefix}"
        return self.generate(prompt, max_new_tokens=max_new_tokens, repeat_input=False)
