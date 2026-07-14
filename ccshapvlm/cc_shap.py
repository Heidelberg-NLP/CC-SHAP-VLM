"""MM-SHAP and CC-SHAP for vision-language models.

``explain`` runs SHAP over the joint image-patch + text-token input of a VLM: it masks
image patches (blanking pixels) and text tokens, scores every masked variant, and returns
per-feature Shapley values. From those we derive:

* **MM-SHAP** (``compute_mm_score``): the share of attribution the model places on text vs.
  image -- a modality-importance score.
* **CC-SHAP** (``measure``): the model's self-consistency between answering and explaining.
  We attribute the answer and the explanation separately, turn each into per-input-token
  ratios (marginalising the prompt scaffolding into SHAP's base value), and compare the two
  profiles; ``score = 1 - cosine_distance`` in ``[-1, 1]``, higher meaning more consistent.

The model-specific mechanics (masking rules, image re-encoding, masked forward passes) are
delegated to the ``VLMAdapter`` hooks; the SHAP orchestration here is model-agnostic.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import spatial, special, stats
from sklearn import metrics

import shap

from ccshapvlm_repro import model_dtype, teacher_forced_scoring

from .config import MAX_NEW_TOKENS
from .models import VLMAdapter

_MAX_EVALS_LADDER = (600, 700, 800, 900)


@dataclass
class CCShapResult:
    mm_score: float
    mm_score_expl: float
    score: float        # 1 - cosine distance: the headline CC-SHAP score
    dist_correl: float
    mse: float
    var: float
    kl_div: float
    js_div: float
    plot_info: dict | int
    prediction_shap: tuple | None = None  # reusable answer attribution for the CoT pass


ZERO_RESULT = CCShapResult(0, 0, 0, 0, 0, 0, 0, 0, plot_info=0)


def compute_mm_score(image_length: int, shap_values) -> float:
    """Textual share of the total attribution mass (MM-SHAP)."""
    image_contrib = np.abs(shap_values.values[0, :image_length, :]).sum()
    text_contrib = np.abs(shap_values.values[0, image_length:, :]).sum()
    return text_contrib / (text_contrib + image_contrib)


def explain(adapter: VLMAdapter, prompt, raw_image, max_new_tokens: int = MAX_NEW_TOKENS,
            num_patches: int | None = None):
    """SHAP values over image patches + text tokens for one VLM forward.

    Returns ``(shap_values, mm_score, num_patches, nb_text_tokens)``. Shape of
    ``shap_values.values``: ``(1, num_input_features, num_output_tokens)``.
    """
    inputs = adapter.encode(prompt, raw_image)
    output_ids = adapter.decode_target_ids(inputs, max_new_tokens)

    nb_text_tokens = inputs.input_ids.shape[1]
    nb_special = adapter.count_special_image_prompt_tokens(inputs.input_ids)
    if num_patches is None:
        num_patches = int(math.ceil(np.sqrt(nb_text_tokens - nb_special)))
    patch_size = inputs.pixel_values.shape[-1] // num_patches
    image_token_ids = torch.tensor(range(-1, -num_patches**2 - 1, -1)).unsqueeze(0)

    def custom_masker(mask, x):
        masked_x = x.clone()
        indices = torch.nonzero(adapter.force_keep_condition(masked_x), as_tuple=False)
        mask[indices[:, 1]] = True
        if nb_special:
            mask[-nb_text_tokens:-nb_text_tokens + nb_special] = True

        image_mask = torch.tensor(mask).unsqueeze(0)
        image_mask[:, -nb_text_tokens:] = True  # leave text tokens for the text mask below
        masked_x[~image_mask] = 0

        text_mask = torch.tensor(mask).unsqueeze(0)
        text_mask[:, :-nb_text_tokens] = True  # image patches already handled
        masked_x[~text_mask] = adapter.text_fill_id
        return masked_x

    score_target = (adapter.teacher_forced_logits if teacher_forced_scoring()
                    else adapter.first_token_logits)

    def score_variants(x):
        with torch.no_grad():
            input_ids = torch.tensor(x[:, -nb_text_tokens:])
            masked_image_ids = torch.tensor(x[:, :-nb_text_tokens])
            result = np.zeros((input_ids.shape[0], output_ids.shape[1]))
            for i in range(input_ids.shape[0]):
                masked_inputs = copy.deepcopy(inputs)
                masked_inputs["input_ids"] = input_ids[i].unsqueeze(0)
                masked_inputs["pixel_values"] = adapter.build_masked_pixel_values(
                    masked_inputs["pixel_values"], raw_image, masked_image_ids[i],
                    num_patches, patch_size, prompt)
                masked_inputs.to("cuda", model_dtype())
                result[i] = score_target(masked_inputs, output_ids)
        return result

    x = torch.cat((image_token_ids, inputs.input_ids), 1).unsqueeze(1)
    shap_values = None
    for max_evals in _MAX_EVALS_LADDER:
        try:
            explainer = shap.Explainer(score_variants, custom_masker, silent=True,
                                       max_evals=max_evals)
            shap_values = explainer(x)[0]
            break
        except ValueError:
            if max_evals == _MAX_EVALS_LADDER[-1]:
                raise

    if len(shap_values.values.shape) == 2:
        shap_values.values = np.expand_dims(shap_values.values, axis=2)

    mm_score = compute_mm_score(image_token_ids.shape[1], shap_values)
    return shap_values, mm_score, num_patches, nb_text_tokens


def _aggregate_explanation(shap_values, len_to_marginalize: int) -> np.ndarray:
    """Per-input-token ratios, folding the trailing scaffolding tokens into the base value."""
    add_to_base = np.abs(shap_values.values[:, -len_to_marginalize:]).sum(axis=1)
    per_output = np.mean(np.abs(shap_values.values[0, -len_to_marginalize:]), axis=0)
    if any(value < 0.01 for value in per_output):
        print("Warning: Some output expl. tokens have very low values. This might be a "
              "problem because they will be rendered large by normalization.")
    ratios = (shap_values.values
              / (np.abs(shap_values.values).sum(axis=1) - add_to_base) * 100)
    return np.mean(ratios, axis=2)[0, :-len_to_marginalize]


def _divergence_scores(ratios_prediction: np.ndarray,
                       ratios_explanation: np.ndarray) -> tuple[float, ...]:
    cosine = spatial.distance.cosine(ratios_prediction, ratios_explanation)
    dist_correl = spatial.distance.correlation(ratios_prediction, ratios_explanation)
    mse = metrics.mean_squared_error(ratios_prediction, ratios_explanation)
    var = (np.sum(((ratios_prediction - ratios_explanation) ** 2 - mse) ** 2)
           / ratios_prediction.shape[0])
    kl_div = stats.entropy(special.softmax(ratios_explanation),
                           special.softmax(ratios_prediction))
    js_div = spatial.distance.jensenshannon(special.softmax(ratios_prediction),
                                            special.softmax(ratios_explanation))
    return cosine, dist_correl, mse, var, kl_div, js_div


def _plot_info(adapter: VLMAdapter, values_prediction, values_explanation,
               num_patches: int, nb_text_tokens_pred: int, nb_text_tokens_expl: int,
               ratios_prediction: np.ndarray, ratios_explanation: np.ndarray) -> dict:
    tokenizer = adapter.text_tokenizer
    input_tokens = values_prediction.data[0].tolist()[-nb_text_tokens_pred:]
    expl_input_tokens = values_explanation.data[0].tolist()[-nb_text_tokens_expl:]
    return {
        "input_tokens": [tokenizer.decode([x], skip_special_tokens=False)
                         for x in input_tokens],
        "ratios_prediction": ratios_prediction.astype(float).round(2).tolist(),
        "expl_input_tokens": [tokenizer.decode([x], skip_special_tokens=False)
                              for x in expl_input_tokens],
        "ratios_explanation": ratios_explanation.astype(float).round(2).tolist(),
        "num_patches_x": num_patches,
    }


def measure(adapter: VLMAdapter, ask_for_prediction, prediction: str, ask_for_expl,
            raw_image, task: str, prediction_shap: tuple | None = None,
            expl_type: str = "post_hoc",
            max_new_tokens: int = MAX_NEW_TOKENS) -> CCShapResult:
    marg_expl = (adapter.post_hoc_expl_phrase(prediction, task) if expl_type == "post_hoc"
                 else adapter.cot_phrase(task))

    if prediction_shap is None:
        prediction_shap = explain(adapter, ask_for_prediction, raw_image, max_new_tokens=1)
    shap_values_pred, mm_score, num_patches, nb_text_tokens_pred = prediction_shap
    shap_values_expl, mm_score_expl, _, nb_text_tokens_expl = explain(
        adapter, ask_for_expl, raw_image, max_new_tokens=max_new_tokens,
        num_patches=num_patches)

    ratios_prediction = _aggregate_explanation(
        shap_values_pred, adapter.marginalize_len(adapter.answer_phrase(task)))
    ratios_explanation = _aggregate_explanation(
        shap_values_expl, adapter.marginalize_len(marg_expl))
    cosine, dist_correl, mse, var, kl_div, js_div = _divergence_scores(
        ratios_prediction, ratios_explanation)

    plot_info = _plot_info(adapter, shap_values_pred, shap_values_expl, num_patches,
                           nb_text_tokens_pred, nb_text_tokens_expl,
                           ratios_prediction, ratios_explanation)
    return CCShapResult(mm_score, mm_score_expl, 1 - cosine, 1 - dist_correl, 1 - mse,
                        1 - var, 1 - kl_div, 1 - js_div, plot_info, prediction_shap)
