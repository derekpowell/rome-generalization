import numpy as np
import pandas as pd
import openai
import configparser
import re 
import time
from  scipy.special import expit, logit


### ----- other helpers

def get_resp_dict(df, item):
    df = df[df["key"] == item]
    df.reset_index(inplace=True)
    string = df["option_mapping"][0]
    a = ast.literal_eval(string)
    res = dict((v,k) for k,v in a.items())

    return(res)

### ------ openai/gpt helpers

def get_gpt_text(response):

    choices = response["choices"]
  
    if type(choices)==list:
        out = [x['text'] for x in choices]
    else: 
        out = choices['text']
    
    return(out)


def make_prompt(query, base_prompt, suffix=""):
    if type(query)==list:
        output = [base_prompt + q + suffix for q in query]
    else:
        output = base_prompt + query + suffix

    return(output)


# def batch_function(x, func, batch_size=20):
    
#     batches = [x[i:i+batch_size] for i in range(0, len(x), batch_size)] # batch into lists of 20
#     # print(batches)
#     out = list()

#     for b in batches:
#         val = func(b)
#         out.append(val)

#     return(out)


def gpt_complete_batch(questions, prompt, model = "text-davinci-003", suffix = "", sleep=0, **kwargs): #  , max_tokens=12, stop=None
  
  # need to add batching into groups of 20 prompts

    response = openai.Completion.create(
        model = model,
        prompt = make_prompt(questions, prompt, suffix),
        **kwargs
    )

    full_text = get_gpt_text(response)

    out = [s.strip() for s in full_text]

    # sleep for sleep seconds (default = 0)
    time.sleep(sleep)

    return(out)    


def gpt_complete(questions, prompt="", suffix = "", model = "text-davinci-003", sleep=0, **kwargs):

    if type(questions)==list:
  
        questions_list = [questions[i:i+20] for i in range(0, len(questions), 20)] # batch into lists of 20
        # should make this a for loop and sleep a bit in between batches
        response_list = [gpt_complete_batch(q, prompt, suffix, model=model, sleep=sleep, **kwargs) for q in questions_list] 
        
        out = list()

        for sub_list in response_list:
            out = out + sub_list

    else:
        out = gpt_complete_batch(questions, prompt, suffix, model=model, **kwargs)

    return(out)


### things for token probs

def get_gpt_logprobs(response):
    x = [r["logprobs"]["token_logprobs"] for r in response["choices"]]
    return(x)


def gpt_token_probs_batch(questions, prompt, suffix = "", model = "text-davinci-003",  sleep=0, **kwargs):
  
  # need to add batching into groups of 20 prompts

    response = openai.Completion.create(
        model=model,
        prompt= make_prompt(questions, prompt, suffix),
        logprobs=1,
        echo=True,
        max_tokens=0,
        **kwargs
    )

    # full_text = get_gpt_text(response)

    out = get_gpt_logprobs(response)
    # sleep for sleep seconds (default = 0)
    time.sleep(sleep)

    return(out)    


def gpt_token_probs(questions, prompt="", suffix = "", model = "text-davinci-003", sleep=0, **kwargs):

    if type(questions)==list:
  
        questions_list = [questions[i:i+20] for i in range(0, len(questions), 20)] # batch into lists of 20
        # should make this a for loop and sleep a bit in between batches
        response_list = [gpt_token_probs_batch(q, prompt, suffix, model=model, sleep=sleep, **kwargs) for q in questions_list] 
        
        out = list()

        for sub_list in response_list:
            out = out + sub_list

    else:
        out = gpt_token_probs_batch(questions, prompt, suffix, model=model, **kwargs)

    return(out)


from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def count_tokens(x):
    return(len(tokenizer(x)["input_ids"]))