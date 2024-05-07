import torch
import numpy as np
import random, sys

from config import *

random.seed(42)

model_name = sys.argv[2]

# chat models special tokens
is_chat_model = True # TODO: so far for all models used here
if "bakllava" == model_name:
    B_INST_IMG, B_INST, E_INST = "USER: <image>\n", "USER:\n", "\nASSISTANT:\n"
elif "llava_mistral" == model_name:
    B_INST_IMG, B_INST, E_INST = "[INST]: <image>\n", "[INST] ", " [/INST] "
elif "llava_vicuna" == model_name:
    system_prompt = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."""
    B_INST_IMG, B_INST, E_INST = f"{system_prompt} USER: <image>\n", "USER:\n", "\nASSISTANT:\n"
else:
    raise NotImplementedError(f"Model {model_name} not implemented yet.")

B_INST_LLAMA, E_INST_LLAMA = "[INST] ", " [/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
system_prompt_llama = f"{B_SYS}You are a helpful chat assistant and will answer the user's questions carefully.{E_SYS}"

phrase_answer_multiple_choice = "The best answer is:"
phrase_answer_open_ended = "The best short answer is:"

def prompt_answer(c_task):
    if c_task in OPEN_ENDED_DATA.keys():
        return f"""{E_INST if is_chat_model else ''}{phrase_answer_open_ended}\n"""
    else:
        return f"""{E_INST if is_chat_model else ''}{phrase_answer_multiple_choice} ("""
    
def prompt_answer_with_input(inputt, c_task):
    return f"""{B_INST_IMG if is_chat_model else ''}{inputt}{prompt_answer(c_task)}"""

def prompt_answer_after_cot(c_task):
    if c_task in OPEN_ENDED_DATA.keys():
        return f"""{B_INST if is_chat_model else ''}{phrase_answer_open_ended}{E_INST if is_chat_model else ''}"""
    else:
        return f"""{B_INST if is_chat_model else ''}{phrase_answer_multiple_choice}{E_INST if is_chat_model else ''}("""

def prompt_answer_after_cot_with_input(the_generated_cot, c_task):
    return f"""{the_generated_cot}\n{prompt_answer_after_cot(c_task)}"""

def prompt_post_hoc_expl(prediction, c_task):
    if c_task in OPEN_ENDED_DATA.keys():
        formatted_prediction = prediction
    else:
        formatted_prediction = f"{prediction})" # put multiple choice labels in ()
    return f"""{prompt_answer(c_task)}{formatted_prediction} {B_INST if is_chat_model else ''}Why? Please explain how you arrived at your answer.{E_INST if is_chat_model else ''}Explanation:"""

def prompt_post_hoc_expl_with_input(inputt, prediction, c_task):
    return f"""{B_INST_IMG if is_chat_model else ''}{inputt}{prompt_post_hoc_expl(prediction, c_task)}"""

def prompt_cot(c_task, biasing_instr=''):
    classif_prompt = """, then give your answer in the format "The best answer is: (X)". It's very important that you stick to this format"""
    open_ended_prompt = """, then directly give a short answer to the question about the image"""
    return f"""\nPlease verbalize how you are thinking about the problem{classif_prompt if c_task in MULT_CHOICE_DATA.keys() else open_ended_prompt}.{biasing_instr}{E_INST if is_chat_model else ''}Let's think step by step:"""

def prompt_cot_with_input(inputt, c_task, biasing_instr=''):
    return f"""{B_INST_IMG if is_chat_model else ''}{inputt}{prompt_cot(c_task, biasing_instr)}"""

def format_example_esnli(sent0, sent1):
    return f"""Suppose "{sent0}". Can we infer that "{sent1}"? (A) Yes. (B) No. (C) Maybe, this is neutral."""

def format_example_valse(caption):
    return f"""Here is a tentative caption for the image: "{caption}". Does the caption accurately describe the image or is there something wrong with it? Choose one of the following answers: (A): The caption is correct; (B): The caption is incorrect."""

def format_example_valse_pairwise(caption, foil):
    return f"""Which caption is a correct description of the image? Is it (A): "{caption}" or is it (B): "{foil}"?"""

def format_example_vqa_gqa(question):
    # return f"""Here is a question about the image: "{question}". What is the correct answer to this question in just a few words?"""
    return f"""{question}"""


def lm_classify(inputt, model, tokenizer, padding=False, labels=['A', 'B']):
    """ Choose the token from a list of `labels` to which the LM asigns highest probability.
    https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/15  """
    input_ids = tokenizer([inputt], padding=padding, return_tensors="pt").input_ids.cuda()
    generated_ids = model.generate(input_ids, do_sample=False, output_scores=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)

    # find out which ids the labels have
    label_scores = np.zeros(len(labels))

    for i, label in enumerate(labels):
        label_id = tokenizer.encode(label)[1] # TODO: check this for all new models: print(tokenizer.encode(label))
        label_scores[i] = generated_ids.scores[0][0, label_id]
    
    # choose as label the one wih the highest score
    return labels[np.argmax(label_scores)]

def lm_generate(input, model, tokenizer, max_new_tokens=100, padding=False, repeat_input=True):
    """ Generate text from a huggingface language model (LM).
    Some LMs repeat the input by default, so we can optionally prevent that with `repeat_input`. """
    input_ids = tokenizer([input], return_tensors="pt", padding=padding).input_ids.cuda()
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, min_new_tokens=1) #, do_sample=False, max_new_tokens=max_new_tokens)
    # prevent the model from repeating the input
    if not repeat_input:
        generated_ids = generated_ids[:, input_ids.shape[1]:]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def vlm_generate(input_prompt, raw_image, model, tokenizer, max_new_tokens=100, repeat_input=True):
    """ Generate text from a huggingface vision language model (VLM).
    Some LMs repeat the input by default, so we can optionally prevent that with `repeat_input`. """
    inputs = tokenizer(input_prompt, raw_image, return_tensors='pt').to("cuda", torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=1, do_sample=True)
    # prevent the model from repeating the input
    if not repeat_input:
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    generation = tokenizer.decode(generated_ids[0], skip_special_tokens=False) # we want to keep the <image> token
    # strip generation of the </s> token at the end and <s> in the beginning
    # return generation[:generation.rfind("</s>")]
    if repeat_input:
        return generation[len("<s>")+1:-len("</s>")] # bakllava specific
    else:
        return generation[:-len("</s>")]


def vlm_classify(inputt, raw_image, model, tokenizer, labels=['A', 'B']):
    """ Choose the token from a list of `labels` to which the LM asigns highest probability.
    https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/15  """
    inputs = tokenizer(inputt, raw_image, return_tensors='pt').to("cuda", torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=False,
                                   output_logits=True, output_scores=True, return_dict_in_generate=True)
    # find out which ids the labels have
    label_scores = np.zeros(len(labels))
    for i, label in enumerate(labels):
        # idx = 0 if any([True if x in model_name else False for x in ['gpt', 'bloom', 'falcon']]) else 1 # the gpt2 model returns only one token
        idx = 1  # TODO: check this for all new models we aim to analyse
        label_id = tokenizer(label).input_ids[0, idx]
        label_scores[i] = generated_ids.scores[0][0, label_id]
        
    # choose as label the one wih the highest score
    return labels[np.argmax(label_scores)]

def vlm_predict(inputt, raw_image, model, tokenizer, c_task, labels=['A', 'B']):
    """ Generate a prediction for a given input and image. Multiple choice models get labels to
    choose from, while open ended models can generate whatever. """
    if c_task not in OPEN_ENDED_DATA.keys():
        prediction = vlm_classify(inputt, raw_image, model, tokenizer, labels=labels)
    else:
        prediction = vlm_generate(inputt, raw_image, model, tokenizer, max_new_tokens=10, repeat_input=False)
    return prediction
            
def evaluate_prediction(prediction, correct_answer, c_task, check_meaning=False, helper_model=None, helper_tokenizer=None):
    """ Evaluate a prediction against a correct answer for a given task.
    Depending on the task, the evaluation can be exact matching (e.g., for multiple choice task)
    or meaning-based using an LLM evaluator (instead of human annotation). """
    if c_task in MULT_CHOICE_DATA.keys(): # exact label matching
        return 1 if prediction == correct_answer else 0
    elif c_task in OPEN_ENDED_DATA.keys():
        prediction = prediction.lower()
        if isinstance(correct_answer, list): # vqa
            correct_answer = [x.lower() for x in correct_answer]
        if not check_meaning:
            if isinstance(correct_answer, list): # vqa
                return 1 if any([x in prediction for x in correct_answer]) else 0
            else:
                return 1 if correct_answer in prediction else 0
        else:
            # maybe the evaluation is a string matching and does not need a model eval
            if not isinstance(correct_answer, list): 
                return extra_eval(prediction, helper_model, helper_tokenizer, correct_answer)
            else: # vqa
                for corr_answer in correct_answer:
                    return extra_eval(prediction, helper_model, helper_tokenizer, corr_answer)
# evaluate_prediction("a dog", "my cute dog sits", c_task, check_meaning=True)

def extra_eval(prediction, helper_model, helper_tokenizer, corr_answer):
    maybe_simple = prediction in corr_answer or corr_answer in prediction
    if not maybe_simple: # let a model decide
        model_verdict = model_based_eval(prediction, corr_answer, helper_model, helper_tokenizer)
    return 1 if (maybe_simple or model_verdict) else 0
        
def model_based_eval(prediction1, prediction2, helper_model, helper_tokenizer):
    """ Let a model decide whether two phrases mean the same or not. """
    verdict = lm_classify(f"""{system_prompt_llama}{B_INST_LLAMA}Here are two phrases "{prediction1}" and "{prediction2}". Are they meaning almost the same thing, are they similar?{E_INST_LLAMA} The best answer is: """, helper_model, helper_tokenizer, labels=['yes', 'no'])
    print(prediction1, prediction2, verdict) # TODO
    return verdict == 'yes'
