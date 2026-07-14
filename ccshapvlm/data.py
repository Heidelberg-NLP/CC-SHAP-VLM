"""Load a task's foils and turn them into the prepared samples the experiment iterates.

``read_data`` normalizes each supported benchmark (VALSE foils, VQA, GQA, NLVR2, the
original FOIL set) into a common ``{id: {caption, image_file, ...}}`` dict; ``prepare_samples``
then formats them. The caption/foil order is shuffled per sample (so the correct answer is
not always "A"), which consumes the global ``random`` stream -- keep the ordering of these
draws stable to keep the assigned labels reproducible across runs.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import MULT_CHOICE_DATA, OPEN_ENDED_DATA, is_multiple_choice, is_open_ended
from .prompts import (format_example_valse, format_example_valse_pairwise,
                      format_example_vqa_gqa)

_MTURK_ACCEPT_THRESHOLD = 2


@dataclass
class Sample:
    formatted_input: str
    correct_answer: str | list[str]
    wrong_answer: str
    image_path: str


@dataclass
class ValseSample:
    """One VALSE item in all three framings the instrument sweeps score."""
    pairwise: str       # shuffled "(A) caption or (B) foil?" question
    caption_only: str   # single-caption question whose correct answer is "A"
    foil_only: str      # single-foil question whose correct answer is "B"
    correct_answer: str
    wrong_answer: str
    image_path: str


def read_data(task: str, foils_path: str, images_path: str, data_root: str) -> dict:
    if task == "vqa":
        return _read_vqa(foils_path, data_root)
    if task in ("gqa", "gqa_balanced"):
        return _read_gqa(foils_path)
    if task == "nlvr2":
        return _read_nlvr2(foils_path, images_path)
    if "original-foil-dataset" in foils_path:
        return _read_foil_dataset(foils_path)
    return json.loads(Path(foils_path).read_text())


def prepare_samples(task: str, data_root: str, num_samples: int) -> list[Sample]:
    if is_multiple_choice(task):
        images_dir, foils_file = MULT_CHOICE_DATA[task]
    elif is_open_ended(task):
        images_dir, foils_file = OPEN_ENDED_DATA[task]
    else:
        raise NotImplementedError(f"Your specified task has no implementation: {task}")

    images_path = f"{data_root}{images_dir}"
    foils_data = read_data(task, f"{data_root}{foils_file}", images_path, data_root)

    builder = _multiple_choice_sample if is_multiple_choice(task) else _open_ended_sample
    samples: list[Sample] = []
    for foil in tqdm(foils_data.values()):
        if len(samples) >= num_samples:
            break
        sample = builder(task, foil, images_path, foils_data)
        if sample is not None:
            samples.append(sample)
    return samples


def prepare_valse_samples(task: str, data_root: str,
                          num_samples: int | None = None) -> list[ValseSample]:
    """VALSE items in pairwise + caption-only + foil-only framings for the sweep scripts."""
    images_dir, foils_file = MULT_CHOICE_DATA[task]
    images_path = f"{data_root}{images_dir}"
    foils_data = read_data(task, f"{data_root}{foils_file}", images_path, data_root)

    samples: list[ValseSample] = []
    for foil in tqdm(foils_data.values()):
        if num_samples is not None and len(samples) >= num_samples:
            break
        sentences = _accepted_sentences(task, foil, foils_data)
        if sentences is None:
            continue
        pairwise, correct, wrong = _pairwise_framing(sentences)
        samples.append(ValseSample(
            pairwise=pairwise,
            caption_only=format_example_valse(sentences[0]),
            foil_only=format_example_valse(sentences[1]),
            correct_answer=correct,
            wrong_answer=wrong,
            image_path=os.path.join(images_path, foil["image_file"]),
        ))
    return samples


def _accepted_sentences(task: str, foil: dict, foils_data: dict) -> list[str] | None:
    """[caption, foil] for an MTURK-accepted item, or None if annotators rejected it."""
    caption_fits = 3 if task == "mscoco" else foil["mturk"]["caption"]
    if caption_fits < _MTURK_ACCEPT_THRESHOLD:
        return None
    if task == "mscoco":
        confounder = random.sample(sorted(foils_data.items()), 1)[0][1]
        return [foil["caption"], confounder["caption"]]
    if task == "plurals":
        return [foil["caption"][0], foil["foil"][0]]
    return [foil["caption"], foil["foil"][0]]


def _pairwise_framing(sentences: list[str]) -> tuple[str, str, str]:
    """Shuffle caption/foil into an A/B question so the answer is not always "A"."""
    if random.choice([0, 1]) == 0:
        return format_example_valse_pairwise(sentences[0], sentences[1]), "A", "B"
    return format_example_valse_pairwise(sentences[1], sentences[0]), "B", "A"


def _multiple_choice_sample(task: str, foil: dict, images_path: str,
                            foils_data: dict) -> Sample | None:
    sentences = _accepted_sentences(task, foil, foils_data)
    if sentences is None:
        return None
    formatted, correct, wrong = _pairwise_framing(sentences)
    return Sample(formatted, correct, wrong, os.path.join(images_path, foil["image_file"]))


def _open_ended_sample(task: str, foil: dict, images_path: str,
                       foils_data: dict) -> Sample:
    correct = foil["answers"] if task == "vqa" else foil["answer"]
    return Sample(format_example_vqa_gqa(foil["caption"]), correct, "impossible to give",
                  os.path.join(images_path, foil["image_file"]))


def _read_foil_dataset(foils_path: str) -> dict:
    """Original FOIL dataset -> our format, recovering the true caption from the foil pair."""
    annotations = json.loads(Path(foils_path).read_text())["annotations"]
    foils_data: dict = {}
    for foil in annotations:
        if foil["foil"] is True:
            image_id = foil["image_id"]
            foils_data[foil["foil_id"]] = {
                "dataset": "FOIL dataset",
                "dataset_idx": foil["foil_id"],
                "original_split": "test",
                "linguistic_phenomena": "noun phrases",
                "image_file": f"COCO_val2014_{str(image_id).zfill(12)}.jpg",
                "caption": foil["caption"].replace(foil["foil_word"], foil["target_word"]),
                "foils": [foil["caption"]],
                "classes": foil["target_word"],
                "classes_foil": foil["foil_word"],
            }
        if len(foils_data) > 1500:
            break
    return foils_data


def _read_vqa(vqa_path: str, data_root: str) -> dict:
    annotations_path = f"{data_root}VQA2.0/v2_mscoco_val2014_annotations.json"
    annotations = json.loads(Path(annotations_path).read_text())["annotations"]
    qa = {ann["question_id"]: ann for ann in annotations}

    questions = json.loads(Path(vqa_path).read_text())["questions"]
    subsample = np.random.choice(len(questions), 300, replace=False)

    foils_data: dict = {}
    for i in subsample:
        sample = questions[i]
        image_id, question_id = sample["image_id"], sample["question_id"]
        foils_data[image_id] = {
            "dataset": "vqa",
            "image_file": f"COCO_val2014_{str(image_id).zfill(12)}.jpg",
            "caption": sample["question"],
            "answer": qa[question_id]["multiple_choice_answer"],
            "answers": [x["answer"] for x in qa[question_id]["answers"]],
        }
    return foils_data


def _read_gqa(gqa_path: str) -> dict:
    gqa_data = json.loads(Path(gqa_path).read_text())
    gqa_data = dict(random.sample(sorted(gqa_data.items()), 300))
    return {
        idx: {
            "dataset": "gqa",
            "dataset_idx": idx,
            "image_file": f'{sample["imageId"]}.jpg',
            "caption": sample["question"],
            "answer": sample["answer"],
        }
        for idx, sample in gqa_data.items()
    }


def _read_nlvr2(nlvr_path: str, images_root: str) -> dict:
    nlvr_data = [json.loads(line) for line in Path(nlvr_path).read_text().splitlines()]
    subsample = np.random.choice(len(nlvr_data), 1500, replace=False)

    foils_data: dict = {}
    for i in subsample:
        sample = nlvr_data[i]
        for side in (0, 1):
            image_file = f'{sample["identifier"][:-2]}-img{side}.png'
            if os.path.isfile(os.path.join(images_root, image_file)):
                foils_data[sample["identifier"]] = {
                    "dataset": "nlvr2",
                    "image_file": image_file,
                    "caption": sample["sentence"],
                    "label": sample["label"],
                }
    return foils_data
