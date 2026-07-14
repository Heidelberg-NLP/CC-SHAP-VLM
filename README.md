# CC-SHAP for VLMs 🖼️
Official code implementation for the paper "Do Vision &amp; Language Decoders use Images and Text equally? How Self-consistent are their Explanations?" accepted at ICLR 2025! 🙌 https://openreview.net/forum?id=lCasyP21Bf

This is follow-up work building on the paper "On Measuring Faithfulness of Natural Language Explanations" https://aclanthology.org/2024.acl-long.329/ that developed CC-SHAP and applied it to LLMs 📃.
Now, we extend to VLMs 🖼️+📃.

## Cite
```bibtex
@inproceedings{parcalabescu2025do,
    title={Do Vision \& Language Decoders use Images and Text equally? How Self-consistent are their Explanations?},
    author={Letitia Parcalabescu and Anette Frank},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=lCasyP21Bf}
}
```

```bibtex
@inproceedings{parcalabescu-frank-2024-measuring,
    title = "On Measuring Faithfulness or Self-consistency of Natural Language Explanations",
    author = "Parcalabescu, Letitia  and
      Frank, Anette",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.329/",
    doi = "10.18653/v1/2024.acl-long.329",
    pages = "6048--6089",
}
```

### Supported Models
1. mPLUG-Owl3 (second paper version) can be run from the `main` branch with the instructions in the next section.
1. BakLLaVA (first paper version) is *experimental* on `main`, because the hf `transformers` version had to be upgraded for supporting mPLUG-Owl3 -- upgrade which brought significant functionality changes. To run the model as it was run for the paper, check out the branch `vicuna-mistral-bakllava-models` and follow the instructions from the `README.md` file there.
1. LLaVA-NeXT-Vicuna (first paper version) same as above.
1. LLaVA-NeXT-Mistral (first paper version) same as above.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (Python 3.12):

```bash
uv sync                                     # create .venv from pyproject.toml
uv pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```

For HuggingFace access and a custom cache location, copy `.env.example` to `.env`
(gitignored) and fill it in:

```dotenv
HF_TOKEN=hf_...                             # only needed for gated models (Llama-2 helper)
HF_CACHE=/path/with/space/for/model/weights # sets HF_HOME so weights don't fill $HOME
```

`.env` is loaded automatically by the `run-*.py` entrypoints (via `hf_env.load_env`)
before any HuggingFace import, so the token and cache directory take effect.

## Models

Models are selected by the second CLI argument and downloaded from the HuggingFace Hub on
first use (keys map to Hub ids in the `MODELS` dict in `config.py`).

| `model` arg | Hub id | Notes |
| --- | --- | --- |
| `mplug-owl3-7b` | `mPLUG/mPLUG-Owl3-7B-240728` | main paper model; fits across the RTX 3090s in fp16 |
| `mplug-owl3-2b` | `mPLUG/mPLUG-Owl3-2B-241014` | smaller mPLUG-Owl3 variant |
| `bakllava`, `llava_mistral`, `llava_vicuna` | see `MODELS` | first-paper models; *experimental* on `main` (see below) |

mPLUG-Owl3 loads via `trust_remote_code`. Its `_decode` needs a one-line fix so the SHAP
path receives output ids instead of decoded text; the fixed file is **vendored** under
[`mplug_owl3/`](mplug_owl3/) (Apache-2.0) and applied to the HuggingFace cache
automatically at load time (`ccshapvlm.models.mplug_owl3.ensure_mplug_remote_code`). To pre-fetch the
weights and apply the patch up front:

```bash
uv run python scripts/setup_mplug.py --model mplug-owl3-7b
```

The `bakllava` / `llava_*` models ran on transformers 4.35 for the first paper version; the
`main` branch upgraded transformers to 4.46 for mPLUG-Owl3, so they are experimental here.
To reproduce them exactly, check out the `vicuna-mistral-bakllava-models` branch.

## Dataset

The paper reads the full VALSE / VQA / GQA sets. For smoke tests we only need a handful
of examples, fetched by `scripts/prepare_sample.py` into `data/`:

```bash
uv run python scripts/prepare_sample.py --num 5
```

This writes the `foil_it` instrument that `config.MULT_CHOICE_DATA` points at:

- `data/foil-it.json` (VALSE `foil-it` annotations, reduced)
- `data/COCO/all_images/*.jpg` (the referenced COCO val2014 images)

For the full benchmarks, download from
[VALSE](https://github.com/Heidelberg-NLP/VALSE),
[VQA](https://visualqa.org/download.html), and
[GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html), then uncomment the
relevant entries in `config.py`.

## Usage

Run from the repository root (so `import shap` resolves to the vendored `shap/`):

```bash
uv run python run-faithfulness.py <task> <model> <num_samples> <save_json> <data_root> [flags]
# e.g.
uv run python run-faithfulness.py foil_it mplug-owl3-7b 5 1 data/
```

Optional flags (defaults come from `config.py`, i.e. the paper's fp16 + sampled decoding
with teacher-forced scoring):

| flag | values | effect |
| --- | --- | --- |
| `--dtype` | `float16` (default) / `float32` | model + image precision; `float32` removes fp16 rounding noise for more stable SHAP scores |
| `--decoding` | `sample` (default) / `greedy` | `greedy` makes generated answers/explanations reproducible across runs |
| `--scoring` | `teacher_forcing` (default) / `efficient` | SHAP target scoring: teacher-forced per-position log-odds vs cheaper first-step logits |

Results are written to `results_json/`. To activate tests individually, edit `TESTS` in
`config.py` (`cc_shap-posthoc` and `cc_shap-cot` must be run together). On `main`,
mPLUG-Owl3 supports the `cc_shap-posthoc` and `cc_shap-cot` tests; the other tests and
models are implemented on the `vicuna-mistral-bakllava-models` branch.

## Credits & licenses
The Shapley value implementation in the `shap/` folder is a modified version of
https://github.com/slundberg/shap . The patched mPLUG-Owl3 model code in `mplug_owl3/` is
Apache-2.0 (see [`mplug_owl3/README.md`](mplug_owl3/README.md) for provenance and the local
modification).

## Disclaimer
This is work in progress. Code and paper will be revised and improved for conference submissions.
