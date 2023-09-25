import pandas as pd
import numpy as np
import json
import time
import random
import pickle


import openai
import configparser
from  scipy.special import expit, logit, logsumexp

from gpthelpers import *

config = configparser.ConfigParser()
config.read("config.ini")
api_key = config.get('Keys','openai_api_key')

openai.api_key = api_key


### --- define funcs

def token_probs_toarraylist(x):
    xnew = []
    for l in x:
        l = [i for i in l if i!=None]
        xnew.append(l)

    return([np.array(i) for i in xnew])

def avg_token_logprob(x):
    tot = np.sum(x) - bias
    N = x.shape[0] - count_tokens(prefix) + 1
    return(tot)

def popularity_logprobs(subj_list):
    resp = gpt_token_probs(subj_list, prefix + " ", model = "text-curie-001")
    ylist = token_probs_toarraylist(resp)
    out = [avg_token_logprob(y) for y in ylist]
    return(out)


### --- define prompt prefix

prefix =  "One example nearly everyone is familiar with is"
prefix_probs = gpt_token_probs(prefix, model = "text-curie-001")
bias = np.sum(token_probs_toarraylist(prefix_probs)[0])

### --- load counterfact data

json_data = [] # your list with json objects (dicts)

with open('counterfact/counterfact.json') as json_file:
   json_data = json.load(json_file)

subjects = list(set([x['requested_rewrite']['subject'] for x in json_data]))
relations = list(set([x['requested_rewrite']['relation_id'] for x in json_data]))
print(len(relations))

### --- get the gpt log probabilities
res_dict = dict()

for r in relations:
    print("scoring popularity for relation:", r)
    subjs = get_subj_by_rid(r)
    scores = popularity_logprobs(subjs)
    res_dict[r] = pd.DataFrame({"subject":subjs, "logprob":scores})


### --- Store data
with open('counterfact/pop-results-curie.pickle', 'wb') as handle:
    pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)