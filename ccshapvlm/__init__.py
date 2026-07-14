"""CC-SHAP for vision-language models.

Measure how self-consistent a VLM is between the answer it gives and the explanation it
offers for that answer (CC-SHAP), alongside its modality reliance (MM-SHAP) and the
prior-work discrete faithfulness tests. See :mod:`ccshapvlm.evaluation` for the entrypoint
loop, :mod:`ccshapvlm.cc_shap` for the method, and :mod:`ccshapvlm.models` for the
per-model adapters.
"""
