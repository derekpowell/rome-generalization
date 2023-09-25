"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets
from util.generate import generate_fast
from util.perplexity import perplexity

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import torch.nn.functional as F

def compute_rewrite_quality_counterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???

    :return: Dictionary containing rewriting metrics
    """
    MODEL_NAME = model.name_or_path

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]
    attribute_prompts = record["attribute_prompts"]
    generation_prompts = record["generation_prompts"]
    
    ## my addition
    low_target_prompts = record["low_target_prompts"][MODEL_NAME] 

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
        attribute_prompts,
        low_target_prompts # addition
    ]
    # Flatten all the evaluated prefixes into one list.
    probs = test_batch_prediction(
        model, tok, list(chain(*prob_prompts)), target_new["str"], target_true["str"]
    )
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
                "attribute_prompts",
                "low_target_prompts" # addition
            ]
        )
    }

    if snips is not None:
        # Gather reference texts
        rel_id = record["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        essence_texts = [
            x["text"]
            for x in snips[rel_id][target_new["id"]]
            if x["name"] == record["requested_rewrite"]["subject"]
        ]
        assert (
            len(consistency_texts) > 0
        ), "Must have consistency texts to evaluate generation"
        gen_stats = test_generation(
            model,
            tok,
            generation_prompts,
            consistency_texts,
            essence_texts,
            vec,
        )
        ret.update(gen_stats)
        
    ## Do my evaluation, creating a dictionary
    
    prompts = [record["subj_const_prompts"][MODEL_NAME][i]["prompt"] for i in range(0,5)]
    # prompts = [x["prompt"][0] for x in record["subj_const_prompts"][MODEL_NAME]]
    orig_gens = [x["gens"][0] for x in record["subj_const_prompts"][MODEL_NAME]]
    
    logprobs_gens = []
    new_target_noappear_logprobs = []
    true_target_noappear_logprobs = []
    bertscore_new = []
    bertscore_true = []
    all_gens = []
    
    new_target_noappear_logprobs_newgens = []
    for p in prompts:
        
        # print("prompt:", p)
        
        probs_noappear_true = logprob_target_not_appear(orig_gens, target_true["str"], p, model, tok)
        # max_idx = torch.max(probs_noappear_true,0).indices.item()
        # least_likely = orig_gens[max_idx]
        # logprobs_gens.append(least_likely)
        # probs_noappear_true = probs_noappear_true[max_idx]
        
        
        probs_noappear_new = logprob_target_not_appear(orig_gens, target_new["str"], p, model, tok)
        true_target_noappear_logprobs.extend(probs_noappear_true.tolist())
        new_target_noappear_logprobs.extend(probs_noappear_new.tolist())
        
        
        
        ## ---- new
        new_gens = generate_fast(model, tok, [p], n_gen_per_prompt = 5, max_out_len = 25) # not great to have this hardcoded
        # print(new_gens)
        
        probs_noappear_new_newgens = logprob_target_not_appear(new_gens, target_new["str"], p, model, tok)
        # print(probs_noappear_new_newgens)

        
        ## use the mean -- expected probability of not including new target token in new generations
        # probs_noappear_new_newgens = torch.mean(np.exp(probs_noappear_new_newgens))
        new_target_noappear_logprobs_newgens.extend(probs_noappear_new_newgens.tolist())
        
        # max_idx = torch.max(probs_noappear_true,0).indices.item()
#         bs_new = calc_bertscore_recall2(gens, target_new["str"])
#         bs_true = calc_bertscore_recall2(gens, target_true["str"])
        
#         bertscore_new.extend(bs_new)
#         bertscore_true.extend(bs_true)
        
#         all_gens.extend(gens)
        
        
    
    bertscore_new = calc_bertscore_recall(model, tok, prompts, target_new["str"])
    bertscore_true = calc_bertscore_recall(model, tok, prompts, target_true["str"])

    my_stats = {
        "subj_gen_sim": calc_subj_gen_similarity(model, tok, prompts, orig_gens),
        "logprob_no_target_true": true_target_noappear_logprobs,
        "logprob_no_target_new": new_target_noappear_logprobs,
        "logprob_no_target_newgens": new_target_noappear_logprobs_newgens,
        "low_prob_gen_text": logprobs_gens,
        "bertscore_new": bertscore_new,
        "bertscore_true": bertscore_true,
        # "model_gen_text": all_gens 
    }

    
    ret.update(my_stats)

    return ret


def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    target_new: str,
    target_true: str,
):
    """ """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    results = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            results[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        results[i] /= cur_len

    return [
        {"target_new": results[i].item(), "target_true": results[i + 1].item()}
        for i in range(0, len(results), 2)
    ]


def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }

    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()


## --- my additions

def test_batch_prediction_single(
    model,
    tok,
    prefixes: typing.List[str],
    target: str,
    pad: bool = True
):
    """ """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    

    if pad:
        prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
        cur_tok = tok(f" {target}")["input_ids"]
        
    else:
        prompt_tok = tok(
        [
            f"{prefix}{suffix}"
            for prefix in prefixes
            for suffix in [target]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
        cur_tok = tok(f"{target}")["input_ids"]
        
    cur_len = len(cur_tok)
    
    with torch.no_grad():
        logits = model(**prompt_tok).logits

    results = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):

        for j in range(cur_len):

            results[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        results[i] /= cur_len

    return [
        {"target": results[0].item()}
    ]

def sentence_similarity_matrix(sentences1, sentences2):

    smodel = SentenceTransformer('all-MiniLM-L6-v2')

    #Compute embedding for both lists
    embeddings1 = smodel.encode(sentences1, convert_to_tensor=True)
    embeddings2 = smodel.encode(sentences2, convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = cos_sim(embeddings1, embeddings2)
    return(cosine_scores)


def avg_sentence_similarity(sentences1, sentences2):
    return(torch.mean(sentence_similarity_matrix(sentences1, sentences2)))


# def generate_sc_text(texts, model, tok, max_new_tokens=15, num_return_sequences = 5):
#     if type(texts) != list:
#         texts = [texts]
#     tok.padding_side = "left"
#     tok.pad_token = tok.eos_token
#     encoding = tok(texts, padding=True, return_tensors='pt').to("cuda")
#     with torch.no_grad():
#         generated_ids = model.generate(**encoding, 
#                                        do_sample=True, 
#                                        temperature=0.7, 
#                                        max_new_tokens = max_new_tokens,
#                                        num_return_sequences = num_return_sequences,
#                                        pad_token_id=tok.eos_token_id
#                                       )

#         generated_texts = tok.batch_decode(
#             generated_ids, skip_special_tokens=True
#         )
        
#     return(generated_texts)


def calc_subj_gen_similarity(model, tok, gen_prompts, orig_gens):
    sims = []
    for i in range(len(gen_prompts)):
        gens = generate_fast(model, tok, [gen_prompts[i]], n_gen_per_prompt = 5, max_out_len = 25) # not great to have this hardcoded
        # gens = [g[len(gen_prompts[i]):] for g in gens] # just use the generated part, not the original prompt
        sentence_similarity = avg_sentence_similarity(gens, orig_gens[i])
        sims.append(sentence_similarity.item())

    mean_sim = sum(sims)/len(sims)
    
    return(mean_sim)


# def calc_subj_gen_similarity_to_target(model, tok, gen_prompts, target):
#     sims = []
#     for i in range(len(gen_prompts)):
#         gens = generate_fast(model, tok, [gen_prompts[i]], n_gen_per_prompt = 5, max_out_len = 25)
#         gens = [g[len(gen_prompts[i]):] for g in gens] # just use the generated part, not the original prompt
#         sentence_similarity = avg_sentence_similarity(gens, orig_gens[i])
#         sims.append(sentence_similarity.item())

#     mean_sim = sum(sims)/len(sims)
    
#     return(mean_sim)

# write a new function to compute max probability of token in certain generations
def encode_token(token:str, tokenizer):
    
    if token[0] != " ": # pad token
        token = " " + token
        
    token_id = tokenizer(token)["input_ids"]
    return(token_id)
    
def token_logits(texts, token, tokenizer, model, start_ind = 0):
    encoding = tokenizer(texts, padding=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        model_out = model(encoding["input_ids"])
        logits = model_out.logits
        logprobs = F.log_softmax(logits, -1)

    token_id = encode_token(token, tokenizer)
    
    return(logprobs[:, start_ind:, token_id])


# def token_max_prob(texts, token, tokenizer, start_ind = 0):
#     token_id = encode_token(token, tokenizer)
#     logits = token_logits(texts, token, tokenizer)
#     # logits = logits[:, start_ind:, token_id]
#     out = torch.max(logits, 1)
    
#     return(out)

def count_tokens(text, tokenizer):
    if type(text)==list:
        assert len(text)==1, "count_tokens() only meant to count tokens of one string at a time"
        
    encoding = tokenizer(text, return_tensors='pt')
    return(len(encoding["input_ids"][0]))


def log1mexp(x):
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    mask = x < 0
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )

def logprob_target_not_appear(texts, target, prompt, model, tokenizer):
    ## what about the probability of it not occuring in the whole string?
    l = token_logits(texts, target, tokenizer, model, count_tokens(prompt, tokenizer))
    return(torch.sum(log1mexp(l), 1))
    

def calc_notarget_prob(model, tok, gen_prompts, target):
    for i in range(len(gen_prompts)):
        gens = generate_fast(model, tok, [gen_prompts[i]], n_gen_per_prompt = 5, max_out_len = 25)
        l = logprob_target_not_appear(gens, target, gen_prompts[i], model, tok)                 
    return(l)



from evaluate import load
bertscore = load("bertscore")

def calc_bertscore_recall(model, tok, gen_prompts, ref):
    sims = []
    for i in range(len(gen_prompts)):
        gens = generate_fast(model, tok, [gen_prompts[i]], n_gen_per_prompt = 5, max_out_len = 25) # not great to have this hardcoded

        references = [ref]*len(gens)
        results = bertscore.compute(predictions=gens, references=references, model_type="distilbert-base-uncased") # "distilbert-base-uncased"

        sims.extend(results["recall"])

    mean_sim = sum(sims)/len(sims) # compute average recall for all prompts
    
    return(mean_sim)

# calc_bertscore_recall(model, tokenizer, [prompt]*2, "baseball")

def calc_bertscore_recall2(gens, ref):
    
    references = [ref]*len(gens)
    results = bertscore.compute(predictions=gens, references=references, model_type="distilbert-base-uncased") # "distilbert-base-uncased"
    
    return(results["recall"])
