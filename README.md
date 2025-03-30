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
1. mPLUG-Owl3 (second paper version) can be run from the main branch with the instructions from the next section.
1. BakLLaVA (first paper version). To run, check out the branch `vicuna-mistral-bakllava-models` and followint the instructions from the `README.md` file there. It is not supported by the current branch because this model requires an older version of hf `transformers` which breaks compatibility with the code written for this (and the following) models.
1. LLaVA-NeXT-Vicuna (first paper version) idem as above.
1. LLaVA-NeXT-Mistral (first paper version) idem as above.

## Installation and running
1. `conda create -n <env-name> python=3.12.1`
2. `pip install -r requirements_pip-mplug-owl3.txt` for installing the required packages with pip for running mPLUG-Owl3 experiments.
3. Download the data from their respective repositories and change the paths in `config.py` accordingly. Data repositories:
  * VALSE 💃: https://github.com/Heidelberg-NLP/VALSE
  * VQA: https://visualqa.org/download.html
  * GQA: https://cs.stanford.edu/people/dorarad/gqa/download.html 
3. To run mPLUG-Owl3, you need to make sure the `_decode` function in `$HF_HOME/modules/transformers_modules/mPLUG/mPLUG-Owl3-7B-240728/eff25bcdc02ff1b513c25f376d761ec1ab6dfa1b/modeling_mplugowl3.py` returns the output ids and not just the text, so update the last lines of that function to:
```python	
    if decode_text:
        output = output[:,input_ids.shape[1]:]
        return self._decode_text(output, tokenizer)
    return output
```
4. Run `run-faithfulness.py` with the following command `python run-faithfulness.py foil_it mplug-owl3-7b 100 0 data/`

## Supported Tests
To activate tests individually, comment the respective elements of `TESTS` in `config.py` (cc_shap-posthoc and cc_shap-cot must be run together). All tests are implemented for the first three models in the other branch. The mPLUG-Owl3 model is only supported for the cc_shap-posthoc and cc_shap-cot tests.

## Credits
The Shapley value implementation in the `shap` folder is a modified version of https://github.com/slundberg/shap .

## Disclaimer
This is work in progress. Code and paper will be revised and improved for conference submissions.
