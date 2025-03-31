import time
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoProcessor, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, AutoConfig

from config import *


def load_models(model_name):
    t1 = time.time()
    if "mplug" in model_name:
        revision_id = "eff25bcdc02ff1b513c25f376d761ec1ab6dfa1b"
        with torch.no_grad():
            model = AutoModel.from_pretrained(MODELS[model_name], attn_implementation='sdpa', torch_dtype=torch.half, revision=revision_id,
            trust_remote_code=True, device_map="auto").eval()
    # elif model_name == "llava_vicuna": # comment this in if you want to use quantisation for llava_vicuna and flash_attention_2
    #     from transformers import BitsAndBytesConfig
    #     # specify how to quantize the model with bitsandbytes
    #     quantization_config = BitsAndBytesConfig(
    #         # load_in_8bit=True,
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.float16,
    #     ) # 8 just load_in_8bit=True,
    #     with torch.no_grad():
    #         model = LlavaNextForConditionalGeneration.from_pretrained(MODELS[model_name], torch_dtype=torch.float16, 
    #             low_cpu_mem_usage=True,
    #             use_flash_attention_2=True,
    #             quantization_config = quantization_config
    #         ) # .to("cuda") not needed for bitsandbytes anymore
    else:
        if model_name == "bakllava":
            ModelClass = LlavaForConditionalGeneration
            revision_id = "a92a28c845fbe89d009f211ce3d0d7aa6d42e948"
        else:
            ModelClass = LlavaNextForConditionalGeneration
        if model_name == "llava_mistral":
            revision_id = "8a7baf5084e1fa437f1fcc58f512fbdb340a2dd9" # hash for the 7b model. For the 2b model it should be 77ab9a6fdb9dae9ce2cd2eda3d32c8ff45ebc7db
        elif model_name == "llava_vicuna":
            revision_id = "89b0f2ea28da2e62d7cfda173a400d2ad82a1c8e"
        with torch.no_grad():
            model = ModelClass.from_pretrained(MODELS[model_name], torch_dtype=torch.float16, 
                low_cpu_mem_usage=True,
                revision=revision_id, #device_map="auto"
                # use_flash_attention_2=True, # comment this in if you want to use flash_attention_2
            ).to("cuda")

    # load tokenizer
    if "mplug" in model_name:
        tokenizer_real = AutoTokenizer.from_pretrained(MODELS[model_name], revision=revision_id)
        processor = model.init_processor(tokenizer_real)
        tokenizer = {"tokenizer": tokenizer_real, "processor": processor}
    else:
        tokenizer = AutoProcessor.from_pretrained(MODELS[model_name])
    print(f"Done loading model and tokenizer after {time.time()-t1:.2f}s.")

    return model, tokenizer