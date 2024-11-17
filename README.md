# CC-SHAP for VLMs 🖼️
Official code implementation for the paper "Do Vision &amp; Language Decoders use Images and Text equally? How Self-consistent are their Explanations?" https://arxiv.org/abs/2404.18624

This is follow-up work building on the paper "On Measuring Faithfulness of Natural Language Explanations" https://arxiv.org/abs/2311.07466  that developed CC-SHAP and applied it to LLMs 📃.
Now, we extend to VLMs 🖼️+📃.

## Cite
```bibtex
@misc{parcalabescu2024vision,
  title={Do Vision \& Language Decoders use Images and Text equally? How Self-consistent are their Explanations?},
  author={Parcalabescu, Letitia and Frank, Anette},
  journal={arXiv preprint arXiv:2404.18624},
  year={2024},
  url = {https://arxiv.org/abs/2404.18624},
  abstract = "Vision and language models (VLMs) are currently the most generally performant architectures on multimodal tasks. Next to their predictions, they can also produce explanations, either in post-hoc or CoT settings. However, it is not clear how much they use the vision and text modalities when generating predictions or explanations. In this work, we investigate if VLMs rely on modalities differently when generating explanations as opposed to when they provide answers. We also evaluate the self-consistency of VLM decoders in both post-hoc and CoT explanation settings, by extending existing tests and measures to VLM decoders. We find that VLMs are less self-consistent than LLMs. The text contributions in VL decoders are much larger than the image contributions across all measured tasks. And the contributions of the image are significantly larger for explanation generations than for answer generation. This difference is even larger in CoT compared to the post-hoc explanation setting. We also provide an up-to-date benchmarking of state-of-the-art VL decoders on the VALSE benchmark, which to date focused only on VL encoders. We find that VL decoders are still struggling with most phenomena tested by VALSE.",
}
```

```bibtex
@article{parcalabescu2023measuring,
  title={On measuring faithfulness or self-consistency of natural language explanations},
  author={Parcalabescu, Letitia and Frank, Anette},
  journal={arXiv preprint arXiv:2311.07466},
  year={2023},
  url      = {https://arxiv.org/abs/2311.07466},
  abstract = "Large language models (LLMs) can explain their own predictions, through post-hoc or Chain-of-Thought (CoT) explanations. However the LLM could make up reasonably sounding explanations that are unfaithful to its underlying reasoning. Recent work has designed tests that aim to judge the faithfulness of either post-hoc or CoT explanations. In this paper we argue that existing faithfulness tests are not actually measuring faithfulness in terms of the models' inner workings, but only evaluate their self-consistency on the output level. The aims of our work are two-fold. i) We aim to clarify the status of existing faithfulness tests in terms of model explainability, characterising them as self-consistency tests instead. This assessment we underline by constructing a Comparative Consistency Bank for self-consistency tests that for the first time compares existing tests on a common suite of 11 open-source LLMs and 5 datasets -- including ii) our own proposed self-consistency measure CC-SHAP. CC-SHAP is a new fine-grained measure (not test) of LLM self-consistency that compares a model's input contributions to answer prediction and generated explanation. With CC-SHAP, we aim to take a step further towards measuring faithfulness with a more interpretable and fine-grained method. Code available at https://github.com/Heidelberg-NLP/CC-SHAP", 
}
```

## Supported Models
1. BakLLaVA (first paper version)
1. LLaVA-NeXT-Vicuna (first paper version)
1. LLaVA-NeXT-Mistral (first paper version)
1. mPLUG-Owl3 (second paper version). To run this model, you need to make sure the `_decode` function in `$HF_HOME/modules/transformers_modules/mPLUG/mPLUG-Owl3-7B-240728/eff25bcdc02ff1b513c25f376d761ec1ab6dfa1b/modeling_mplugowl3.py` returns the output ids and not just the text, so update the last lines of that function to:
```python	
    if decode_text:
        output = output[:,input_ids.shape[1]:]
        return self._decode_text(output, tokenizer)
    return output
```

## Supported Tests
To activate tests individually, comment the respective elements of `TESTS` in `config.py` (cc_shap-posthoc and cc_shap-cot must be run together). All tests are implemented for the first three models. The mPLUG-Owl3 model is only supported for the cc_shap-posthoc and cc_shap-cot tests.


## Installation
See `requirements_pip-mplug-owl3.txt` for installing the required packages with pip. This was used to run the mPLUG-Owl3 experiments and should work for the first three models too.
The experiments for the other 3 models were run with the installation from `requirements_conda.txt`.

## Credits
The Shapley value implementation in the `shap` folder is a modified version of https://github.com/slundberg/shap .

## Disclaimer
This is work in progress. Code and paper will be revised and improved for conference submissions.
