"""Static configuration: model registry, task datasets, labels, and runtime defaults."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

MAX_NEW_TOKENS = 100

# Runtime defaults, overridable per run via the run-* flags (--dtype/--decoding/--scoring).
# Paper behavior is fp16 weights with sampled decoding; switch to float32 + greedy for
# more stable, cross-run-comparable scores.
DTYPE = "float16"     # "float16" | "float32"
DECODING = "sample"   # "sample" | "greedy"

# SHAP target scoring: teacher-forced per-position log-odds (text CC-SHAP semantics) or the
# cheaper first-step logits.
SCORING = "teacher_forcing"  # "teacher_forcing" | "efficient"

TESTS: tuple[str, ...] = (
    "cc_shap-posthoc",
    "cc_shap-cot",
    # "atanasova_counterfactual",
    # "turpin",
    # "lanham",
)

MODELS = {
    "bakllava": "llava-hf/bakLlava-v1-hf",
    "llava_mistral": "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava_vicuna": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "mplug-owl3-2b": "mPLUG/mPLUG-Owl3-2B-241014",
    "mplug-owl3-7b": "mPLUG/mPLUG-Owl3-7B-240728",
}

# Pinned remote-code revisions. The 7B hash is the one the paper ran and the one the
# vendored modeling_mplugowl3.py patch is derived from.
MPLUG_REVISIONS = {
    "mplug-owl3-7b": "eff25bcdc02ff1b513c25f376d761ec1ab6dfa1b",
    "mplug-owl3-2b": None,
}

# Fixed remote-code revisions for the LLaVA-family checkpoints.
LLAVA_REVISIONS = {
    "bakllava": "a92a28c845fbe89d009f211ce3d0d7aa6d42e948",
    "llava_mistral": "8a7baf5084e1fa437f1fcc58f512fbdb340a2dd9",
    "llava_vicuna": "89b0f2ea28da2e62d7cfda173a400d2ad82a1c8e",
}

HELPER_MODEL = "llama2-13b-chat"

LABELS = {
    "esnli": ["A", "B", "C"],
    "binary": ["A", "B"],
}

# task -> (images subdir, foils json), relative to the data root. Only foil_it ships with a
# prepared data sample; the other VALSE instruments are supported (read by read_data) but
# left commented until their benchmark files are placed under the data root.
MULT_CHOICE_DATA = {
    "foil_it": ["COCO/all_images/", "foil-it.json"],
    # "existence": ["foil-benchmark-old/counting++/images/",
    #               "foil-benchmark/existence/existence_benchmark.test_mturk.json"],
    # "plurals": ["foil-benchmark/plurals/test_images/",
    #             "foil-benchmark/plurals/plurals_test_mturk.json"],
    # "counting_hard": ["foil-benchmark-old/counting++/images/",
    #                   "foil-benchmark/counting_hard/visual7w_counting.hard.test_mturk.json"],
    # "counting_small": ["foil-benchmark-old/counting++/images/",
    #                    "foil-benchmark/counting/visual7w_counting.small-quantities.test_mturk.json"],
    # "counting_adv": ["foil-benchmark-old/counting++/images/",
    #                  "foil-benchmark/counting_adversarial/visual7w_counting.adversarial.test_mturk.json"],
    # "relations": ["foil-benchmark/relations/test_images/",
    #               "foil-benchmark/relations/relations_test_mturk.json"],
    # "action_replace": ["foil-benchmark/actions/images_512/",
    #                    "foil-benchmark/actions/action_replace/action_replace_test_mturk.json"],
    # "actant_swap": ["foil-benchmark/actions/images_512/",
    #                 "foil-benchmark/actions/actant_swap/actant_swap_test_mturk.json"],
    # "coref": ["foil-benchmark/coref/release_too_many_is_this_in_color/images/",
    #           "foil-benchmark/coref/coref_test_visdial_train_mturk.json"],
    # "coref_hard": ["foil-benchmark/coref/release_v18/test_images/",
    #                "foil-benchmark/coref/coref_test_hard_mturk.json"],
    # "mscoco": ["COCO/all_images/", "foil-benchmark/orig_foil/foil_it_test_mturk.json"],
}

# Open-ended VQA tasks (answers scored by substring / meaning match). Readers live in
# ccshapvlm.data; enable a task by uncommenting it and providing its files.
OPEN_ENDED_DATA: dict[str, list[str]] = {
    # "vqa": ["COCO/all_images/", "VQA2.0/v2_OpenEnded_mscoco_val2014_questions.json"],
    # "gqa": ["GQA/images/", "GQA/val_all_questions.json"],
    # "gqa_balanced": ["GQA/images/", "GQA/val_balanced_questions.json"],
}


def greedy_generation() -> bool:
    """Decode answers/explanations greedily instead of sampling (``--decoding``).

    Paper behavior samples (``do_sample=True``), which is irreproducible run to run; greedy
    decoding makes the generated text -- and the scores computed on it -- comparable.
    """
    return str(DECODING).lower() == "greedy"


def teacher_forced_scoring() -> bool:
    """Score SHAP targets by teacher-forced log-odds (default) vs first-step logits."""
    return str(SCORING).lower() == "teacher_forcing"


def model_dtype() -> torch.dtype:
    """Torch dtype for model weights and image inputs (``--dtype``; default fp16).

    fp16 matmul rounding is non-associative and, amplified across the SHAP explainer's many
    masked forward passes, shifts the approximate SHAP ratios run to run; fp32 removes that.
    """
    import torch

    return {"float32": torch.float32, "fp32": torch.float32}.get(
        str(DTYPE).lower(), torch.float16)


def is_open_ended(task: str) -> bool:
    return task in OPEN_ENDED_DATA


def is_multiple_choice(task: str) -> bool:
    return task in MULT_CHOICE_DATA
