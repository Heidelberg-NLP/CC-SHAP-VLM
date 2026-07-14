"""LLaVA-family adapters: BakLLaVA and LLaVA-NeXT (Mistral / Vicuna).

All three take instruction-token strings and a ``(text, image)`` processor call, and differ
only in literals: their instruction tokens, the special token ids SHAP must keep, the id
that replaces masked text, and whether an image patch is blanked in pixel space (BakLLaVA)
or by re-encoding a zeroed image array (LLaVA-NeXT).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from transformers import (AutoProcessor, LlavaForConditionalGeneration,
                          LlavaNextForConditionalGeneration)

from ccshapvlm_repro import greedy_generation, model_dtype

from ..config import LLAVA_REVISIONS, MAX_NEW_TOKENS, MODELS
from .base import Prompt, VLMAdapter

_VICUNA_SYSTEM = ("A chat between a curious human and an artificial intelligence assistant. "
                  "The assistant gives helpful, detailed, and polite answers to the human's "
                  "questions.")


@dataclass(frozen=True)
class _Spec:
    b_inst_img: str
    b_inst: str
    e_inst: str
    extra_keep_id: int
    text_fill_id: int
    patches_in_place: bool
    next_generation: bool  # LLaVA-NeXT vs. the original LLaVA architecture


_SPECS = {
    "bakllava": _Spec("USER: <image>\n", "USER:\n", "\nASSISTANT:\n",
                      extra_keep_id=28705, text_fill_id=583,
                      patches_in_place=True, next_generation=False),
    "llava_mistral": _Spec("[INST]: <image>\n", "[INST] ", " [/INST] ",
                           extra_keep_id=28705, text_fill_id=583,
                           patches_in_place=False, next_generation=True),
    "llava_vicuna": _Spec(f"{_VICUNA_SYSTEM} USER: <image>\n", "USER:\n", "\nASSISTANT:\n",
                          extra_keep_id=29871, text_fill_id=903,
                          patches_in_place=False, next_generation=True),
}


class LlavaFamilyAdapter(VLMAdapter):
    def __init__(self, name: str, model, processor, spec: _Spec):
        self.name = name
        self.model = model
        self._processor = processor
        self._spec = spec
        self.B_INST_IMG = spec.b_inst_img
        self.B_INST = spec.b_inst
        self.E_INST = spec.e_inst
        self.text_fill_id = spec.text_fill_id

    @classmethod
    def load(cls, name: str) -> "LlavaFamilyAdapter":
        spec = _SPECS[name]
        model_class = (LlavaNextForConditionalGeneration if spec.next_generation
                       else LlavaForConditionalGeneration)
        with torch.no_grad():
            model = model_class.from_pretrained(
                MODELS[name], torch_dtype=model_dtype(), low_cpu_mem_usage=True,
                revision=LLAVA_REVISIONS[name]).to("cuda")
        processor = AutoProcessor.from_pretrained(MODELS[name])
        return cls(name, model, processor, spec)

    def build_answer_input(self, formatted_sample: str, task: str) -> Prompt:
        return f"{self.B_INST_IMG}{formatted_sample}{self.answer_phrase(task)}"

    def build_post_hoc_expl_input(self, formatted_sample: str, prediction: str,
                                  task: str) -> Prompt:
        return (f"{self.B_INST_IMG}{formatted_sample}"
                f"{self.post_hoc_expl_phrase(prediction, task)}")

    def build_cot_input(self, formatted_sample: str, task: str,
                        biasing_instr: str = "") -> Prompt:
        return f"{self.B_INST_IMG}{formatted_sample}{self.cot_phrase(task, biasing_instr)}"

    def build_answer_after_cot_input(self, generated_cot: str, task: str,
                                     biasing_instr: str = "",
                                     inputt: str | None = None) -> Prompt:
        return f"{generated_cot}\n{self.answer_after_cot_phrase(task)}"

    def _encode(self, prompt: Prompt, image):
        return self._processor(prompt, image, return_tensors="pt").to("cuda", model_dtype())

    def generate(self, prompt: Prompt, image, max_new_tokens: int = MAX_NEW_TOKENS,
                 repeat_input: bool = True, skip_special_tokens: bool = False) -> str:
        inputs = self._encode(prompt, image)
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, min_new_tokens=1,
            do_sample=not greedy_generation())
        if not repeat_input:
            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        generation = self._processor.decode(generated_ids[0],
                                            skip_special_tokens=skip_special_tokens)
        if repeat_input:
            return generation[len("<s>") + 1:-len("</s>")]
        return generation[:-len("</s>")]

    def classify(self, prompt: Prompt, image, labels: list[str]) -> str:
        inputs = self._encode(prompt, image)
        generated = self.model.generate(
            **inputs, max_new_tokens=1, min_new_tokens=1, do_sample=False,
            output_logits=True, output_scores=True, return_dict_in_generate=True)
        scores = np.zeros(len(labels))
        for i, label in enumerate(labels):
            label_id = self._processor(label, return_tensors="pt").input_ids[0, 1]
            scores[i] = generated.scores[0][0, label_id]
        return labels[int(np.argmax(scores))]

    @property
    def text_tokenizer(self):
        return self._processor

    def marginalize_len(self, text: str) -> int:
        return self._processor(text, return_tensors="pt").input_ids.shape[1] - 2

    def encode(self, prompt: Prompt, image):
        return self._encode(prompt, image)

    def decode_target_ids(self, inputs, max_new_tokens: int):
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, min_new_tokens=1,
            do_sample=not greedy_generation())
        output_ids = outputs[:, inputs.input_ids.shape[1]:].to("cpu")
        inputs.to("cpu")
        return output_ids

    def count_special_image_prompt_tokens(self, input_ids) -> int:
        return 0

    def force_keep_condition(self, masked_x):
        return ((masked_x == 1) | (masked_x == 32000)
                | (masked_x == self._spec.extra_keep_id))

    def build_masked_pixel_values(self, base_pixel_values, raw_image, masked_ids_row,
                                  num_patches: int, patch_size: int, prompt: Prompt):
        if self._spec.patches_in_place:
            for k in range(masked_ids_row.shape[0]):
                if masked_ids_row[k] == 0:
                    m, n = k // num_patches, k % num_patches
                    base_pixel_values[:, :, m * patch_size:(m + 1) * patch_size,
                                      n * patch_size:(n + 1) * patch_size] = 0
            return base_pixel_values

        raw_image_array = copy.deepcopy(np.array(raw_image))
        for k in range(masked_ids_row.shape[0]):
            if masked_ids_row[k] == 0:
                m, n = k // num_patches, k % num_patches
                raw_image_array[m * patch_size:(m + 1) * patch_size,
                                n * patch_size:(n + 1) * patch_size, :] = 0
        return self._processor(prompt, raw_image_array, return_tensors="pt").pixel_values

    def first_token_logits(self, masked_inputs, output_ids):
        out = self.model.generate(
            **masked_inputs, max_new_tokens=1, do_sample=False,
            output_logits=True, output_scores=True, return_dict_in_generate=True)
        logits = out.logits[0].detach().cpu().numpy()
        return logits[0, output_ids]

    def _extend_token_aligned_inputs(self, masked_inputs, n_new_tokens: int) -> None:
        mask = masked_inputs.get("attention_mask")
        if mask is not None:
            masked_inputs["attention_mask"] = torch.cat(
                [mask, mask.new_ones((mask.shape[0], n_new_tokens))], dim=-1)
