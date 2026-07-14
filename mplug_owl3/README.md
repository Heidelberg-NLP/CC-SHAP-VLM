# mPLUG-Owl3 patched model code

The code in this directory is third-party code, **not** covered by the license in the
repository root.

- **Origin:** [`mPLUG/mPLUG-Owl3-7B-240728`](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-240728)
  (revision `eff25bcdc02ff1b513c25f376d761ec1ab6dfa1b`), by X-PLUG / Alibaba.
- **License:** Apache-2.0 (full text in [`LICENSE`](./LICENSE)).
- **Modification:** `modeling_mplugowl3.py` — the `_decode` method originally sliced off
  the prompt ids unconditionally (`output = output[:, input_ids.shape[1]:]`). On the SHAP
  path (`decode_text=False`, which passes `return_dict_in_generate=True`) `output` is a
  `GenerateOutput`, so that slice crashed / discarded the logits the explainer reads. The
  slice is now guarded by `if decode_text:` and callers strip the prompt ids themselves.
  This is the fix the upstream README asks users to apply by hand; here it is hosted and
  applied automatically.

The model **weights** (`model.safetensors`, ~16 GB) and the remaining remote-code files
are **not** vendored. `scripts/setup_mplug.py` downloads the full snapshot into the
HuggingFace cache and overlays this patched `modeling_mplugowl3.py` onto it, so
`trust_remote_code=True` loads the fixed version. `ccshapvlm.models.mplug_owl3.ensure_mplug_remote_code`
does the same automatically at load time.
