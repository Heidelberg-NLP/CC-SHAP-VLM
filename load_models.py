import torch
import time
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration

from config import *


def load_models(model_name):
    t1 = time.time()
    if model_name == "llava_vicuna":
        from transformers import BitsAndBytesConfig
        # specify how to quantize the model with bitsandbytes
        quantization_config = BitsAndBytesConfig(
            # load_in_8bit=True,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ) # 8 just load_in_8bit=True,

        with torch.no_grad():
            model = LlavaNextForConditionalGeneration.from_pretrained(MODELS[model_name], torch_dtype=torch.float16, 
                low_cpu_mem_usage=True,
                use_flash_attention_2=False, # set to true for speedups and install flash-attn
                quantization_config = quantization_config
            ) # .to("cuda") not needed for bitsandbytes anymore
        revision_id = "0524afe4453163103dcefe78eb0a58b3f6424eac"
    else:
        if model_name == "bakllava":
            ModelClass = LlavaForConditionalGeneration
            # We need to use revision numbers. Not doing so, will make `from_pretrained` load the newest model commit from the hub, which is unfortunately not backwards compatible with the transformers version used in this repo / branch
            revision_id = "f038f156966ff4d24078b260e9e9761fd480d325"
        elif model_name == "llava_mistral":
            ModelClass = LlavaNextForConditionalGeneration
            revision_id = "a1d521368f8d353afa4da2ed2bb1bf646ef1ff5f"
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        with torch.no_grad():
            model = ModelClass.from_pretrained(MODELS[model_name], torch_dtype=torch.float16, 
                low_cpu_mem_usage=True, # device_map="auto",
                use_flash_attention_2=False, # set to true for speedups and install flash-attn
                revision=revision_id,
            ).to("cuda")
    tokenizer = AutoProcessor.from_pretrained(MODELS[model_name], revision=revision_id)
    print(f"Done loading model and tokenizer after {time.time()-t1:.2f}s.")

    return model, tokenizer