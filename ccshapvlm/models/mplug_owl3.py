"""mPLUG-Owl3 adapter.

mPLUG-Owl3 takes chat message lists (with a ``<|image|>`` marker), is queried through a
processor that expects ``tokenizer``/``decode_text`` alongside the tensors, and prepends a
fixed block of special image-prompt tokens that must never be masked by SHAP.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from ..config import (MAX_NEW_TOKENS, MODELS, MPLUG_REVISIONS, greedy_generation,
                      is_open_ended, model_dtype)
from ..prompts import PHRASE_ANSWER_MULTIPLE_CHOICE, PHRASE_ANSWER_OPEN_ENDED
from .base import Prompt, VLMAdapter

_PATCH_FILE = "modeling_mplugowl3.py"
_PATCH_SOURCE = Path(__file__).resolve().parents[2] / "mplug_owl3" / _PATCH_FILE

# the last ``<|image|>``-region token (1805) plus its trailing 4 scaffold tokens.
_IMAGE_PROMPT_MARKER = 1805
_KEEP_TOKEN_IDS = [151644, 151645]  # <|im_start|>, <|im_end|>


def _overlay(dest: Path, patched: bytes) -> None:
    if dest.is_file() and dest.read_bytes() == patched:
        return
    if dest.is_symlink():
        dest.unlink()  # replace the cache symlink with a real file, don't edit the blob
    dest.write_bytes(patched)


def ensure_mplug_remote_code(model_id: str, revision: str | None = None) -> None:
    """Overlay the vendored patched ``modeling_mplugowl3.py`` onto the HF cache.

    ``trust_remote_code`` copies the model's ``.py`` files from the snapshot into
    ``$HF_HOME/modules`` and imports them, so we patch both the snapshot source (what
    transformers copies from) and any already-materialized module copy.
    """
    from huggingface_hub import hf_hub_download

    patched = _PATCH_SOURCE.read_bytes()
    _overlay(Path(hf_hub_download(model_id, _PATCH_FILE, revision=revision)), patched)

    modules_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    for module_copy in (modules_root / "modules").rglob(_PATCH_FILE):
        _overlay(module_copy, patched)


def _user_content(inputt: str) -> str:
    return f"<|image|>\n        {inputt}."


class MplugOwl3Adapter(VLMAdapter):
    text_fill_id = 220

    COT_LEAD = "I will verbalize how I am thinking about the problem"
    COT_CLASSIF = (', then give my answer in the format "The best answer is: (X)". '
                   "It's very important that you stick to this format")

    def __init__(self, name: str, model, tokenizer, processor):
        self.name = name
        self.model = model
        self._tokenizer = tokenizer
        self.processor = processor

    @classmethod
    def load(cls, name: str) -> "MplugOwl3Adapter":
        revision = MPLUG_REVISIONS.get(name)
        ensure_mplug_remote_code(MODELS[name], revision)
        with torch.no_grad():
            model = AutoModel.from_pretrained(
                MODELS[name], attn_implementation="sdpa", torch_dtype=model_dtype(),
                revision=revision, trust_remote_code=True, device_map="auto").eval()
        tokenizer = AutoTokenizer.from_pretrained(MODELS[name], revision=revision)
        processor = model.init_processor(tokenizer)
        return cls(name, model, tokenizer, processor)

    def _answer_open_or_mc(self, task: str) -> str:
        return PHRASE_ANSWER_OPEN_ENDED if is_open_ended(task) else PHRASE_ANSWER_MULTIPLE_CHOICE

    def build_answer_input(self, formatted_sample: str, task: str) -> Prompt:
        return [{"role": "user", "content": _user_content(formatted_sample)},
                {"role": "assistant", "content": f"{self._answer_open_or_mc(task)} ("}]

    def build_post_hoc_expl_input(self, formatted_sample: str, prediction: str,
                                  task: str) -> Prompt:
        return [{"role": "user", "content": _user_content(formatted_sample)},
                {"role": "assistant", "content": self._answer_open_or_mc(task)}]

    def build_cot_input(self, formatted_sample: str, task: str,
                        biasing_instr: str = "") -> Prompt:
        return [{"role": "user", "content": _user_content(formatted_sample)},
                {"role": "assistant", "content": self.cot_phrase(task, biasing_instr)}]

    def build_answer_after_cot_input(self, generated_cot: str, task: str,
                                     biasing_instr: str = "",
                                     inputt: str | None = None) -> Prompt:
        assistant = (f"{self.cot_phrase(task, biasing_instr)} {generated_cot}\n"
                     f"{self.answer_after_cot_phrase(task)}")
        return [{"role": "user", "content": _user_content(inputt)},
                {"role": "assistant", "content": assistant}]

    # mPLUG decodes to text itself; repeat_input / skip_special_tokens do not apply.
    def generate(self, prompt: Prompt, image, max_new_tokens: int = MAX_NEW_TOKENS,
                 repeat_input: bool = True, skip_special_tokens: bool = False) -> str:
        inputs = self.processor(copy.deepcopy(prompt), images=[image], videos=None)
        inputs.to("cuda")
        inputs.update({"tokenizer": self._tokenizer, "max_new_tokens": max_new_tokens,
                       "decode_text": True})
        return self.model.generate(**inputs, min_new_tokens=1,
                                   do_sample=not greedy_generation())[0]

    def classify(self, prompt: Prompt, image, labels: list[str]) -> str:
        inputs = self.processor(copy.deepcopy(prompt), images=[image], videos=None)
        inputs.to("cuda")
        inputs.update({"tokenizer": self._tokenizer, "max_new_tokens": MAX_NEW_TOKENS,
                       "decode_text": False})
        generated = self.model.generate(
            **inputs, min_new_tokens=1, do_sample=False, output_logits=True,
            output_scores=True, return_dict_in_generate=True)
        scores = np.zeros(len(labels))
        for i, label in enumerate(labels):
            label_id = self._tokenizer(label).input_ids[0]
            scores[i] = generated.scores[0][0, label_id]
        return labels[int(np.argmax(scores))]

    @property
    def text_tokenizer(self):
        return self._tokenizer

    def marginalize_len(self, text: str) -> int:
        return len(self._tokenizer(text))

    def encode(self, prompt: Prompt, image):
        return self.processor(copy.deepcopy(prompt), images=[image], videos=None)

    def decode_target_ids(self, inputs, max_new_tokens: int):
        query = copy.deepcopy(inputs)
        query.to("cuda")
        query.update({"tokenizer": self._tokenizer, "max_new_tokens": max_new_tokens,
                      "decode_text": False})
        outputs = self.model.generate(**query, min_new_tokens=1,
                                      do_sample=not greedy_generation())
        output_ids = outputs[:, inputs.input_ids.shape[1]:].to("cpu")
        inputs.to("cpu")
        return output_ids

    def count_special_image_prompt_tokens(self, input_ids) -> int:
        return int(np.where(input_ids[0] == _IMAGE_PROMPT_MARKER)[0][-1] + 4)

    def force_keep_condition(self, masked_x):
        return torch.isin(masked_x, torch.tensor(_KEEP_TOKEN_IDS))

    def build_masked_pixel_values(self, base_pixel_values, raw_image, masked_ids_row,
                                  num_patches: int, patch_size: int, prompt: Prompt):
        raw_image_array = copy.deepcopy(np.array(raw_image))
        for k in range(masked_ids_row.shape[0]):
            if masked_ids_row[k] == 0:
                m, n = k // num_patches, k % num_patches
                raw_image_array[m * patch_size:(m + 1) * patch_size,
                                n * patch_size:(n + 1) * patch_size, :] = 0
        pil_image = Image.fromarray(raw_image_array).convert("RGB")
        return self.processor(copy.deepcopy(prompt), images=[pil_image],
                              videos=None)["pixel_values"]

    def first_token_logits(self, masked_inputs, output_ids):
        masked_inputs.update({"tokenizer": self._tokenizer, "max_new_tokens": 1,
                              "decode_text": False})
        out = self.model.generate(**masked_inputs, do_sample=False, output_logits=True,
                                  output_scores=True, return_dict_in_generate=True)
        logits = out.logits[0].detach().cpu().numpy()
        return logits[0, output_ids]

    def _extend_token_aligned_inputs(self, masked_inputs, n_new_tokens: int) -> None:
        # media_offset counts image tokens seen so far per position; appended target tokens
        # sit after the image, so its count is constant -- repeat the last row.
        media_offset = masked_inputs["media_offset"]
        masked_inputs["media_offset"] = torch.cat(
            [media_offset, media_offset[:, -1:, :].expand(-1, n_new_tokens, -1)], dim=1)
