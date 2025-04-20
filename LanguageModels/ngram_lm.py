import logging
import nltk
from nltk.util import ngrams
nltk.download('punkt_tab')

from typing import Dict, Set
from collections import Counter 
from random import choices
from tqdm import tqdm

from typing import List, Dict 

## 
##  Author: Xavier
## 
##  This file is part of the python code for the course Introduction to Natural Language Processing. 
## 
##  Released April 10, 2025. 
## 

start_symbol = "_START_"
stop_symbol = "_STOP_"


def count_ngrams_up_to(n_max: int, texts: List[str], tokenizer=nltk.word_tokenize) -> Dict[tuple, Counter]:
    """Given a collection of texts, computes counts of ngrams up to length n_max. 
    
    The counts are returned as a dictionary that maps ngram contexts to counts of the last symbol of each ngram. 

    For instance, if we see ngram (a, b, c) 1 time, and (a, b, d) 2 times we will have the context is (a, b) and the
    counter for this context gives a count of 1 for c and a count of 2 for d. 

        (a, b) -> {c: 1, d: 2}
    
    Therefore all unigrams counts are in the empty context (). The bigram counts are found in context of length 1, 
    trigram counts are found in context of length 2, and so on.     
    
    """
    counts = {} # dict from context to a counter of next symbol
    for text in texts:
        tokens = tokenizer(text) + [stop_symbol]
        for n in range(n_max):
            starts = [start_symbol] * n
            for ngram in ngrams(starts + tokens, n+1):
                context = ngram[:-1]
                end_symbol = ngram[-1]
                counts.setdefault(context, Counter()).update([end_symbol])
    return counts


class NGramLanguageModel:
    """ 
    An NGram Language model with Katz back-off discount. 

    n is the order of the ngram model, i.e n=3 for a trigram model

    ngram_counts is a dictionary of word counts for each ngram context. 

    back_off_discount is the discount value for Katz back-off. A None value 
       indicates no-back off, the counts are used without any smoothing,. 

    """
    def __init__(self, n: int, ngram_counts: Dict[tuple, Counter], back_off_discount: float):
        self.n = n 
        self.back_off_discount = back_off_discount
        self.ngram_counts = ngram_counts
        self.vocab = set(self.ngram_counts.get((), {}).keys())
    
    def p_next_word(self, tokens: tuple, top: int = None, n=None, _vocab: Set[str] = None) -> dict[str, float]:
        """Returns the probability distribution over the next word given the tokens. """
        logger = logging.getLogger('p_next_word')
        if n is None:
            n = self.n 
        elif n == 0:
            return {}
        ctx_len = n-1   # len of context is the order of the ngram model minus one 
        if ctx_len == 0: 
            ctx = ()  # take empty context
            ctx_counts = self.ngram_counts.get(ctx)   
        else:
            if len(tokens) < ctx_len:
                starts = [start_symbol] * (ctx_len - len(tokens))
                tokens = starts + list(tokens)            
            ctx = tuple(tokens[-ctx_len:])
            assert(len(ctx)==ctx_len)
            ctx_counts = self.ngram_counts.get(ctx, Counter()) # take counts of context, or an empty Counter if context does not exist
        if _vocab is None: 
            _vocab = self.vocab
        total_count = sum([ctx_counts[w] for w in _vocab])
        logger.debug(f"context_length={ctx_len} context={ctx} total_count={total_count} observed_words={len(ctx_counts)}")

        if self.back_off_discount:
            _vocab_observed = set(_vocab).intersection(set(ctx_counts.keys()))
            _vocab_unobserved = set(_vocab).difference(_vocab_observed)
            word_prob_observed = [(word, (ctx_counts.get(word, 0) - self.back_off_discount) / total_count) for word in _vocab_observed]
            total_discount = self.back_off_discount*len(_vocab_observed)
            if total_count == 0:
                mass_discount = 1
            else: 
                mass_discount = total_discount / total_count
            logger.debug(f"backing-off to context len {n-1} for {len(_vocab_unobserved)} unobserved words")
            word_prob_unobserved = [(w, p*mass_discount) for w, p in self.p_next_word(tokens, n=n-1, _vocab=_vocab_unobserved).items()]
            logger.debug(f"got this from recursive: {word_prob_unobserved}")
            word_prob = word_prob_observed + word_prob_unobserved
        elif total_count == 0:
            word_prob = [(word, 0) for word in _vocab]
        else:
            word_prob = [(word, ctx_counts.get(word, 0) / total_count) for word in _vocab]
        word_prob = sorted(word_prob, key=lambda wp: wp[1], reverse=True)
        if top is not None:
            word_prob = word_prob[:top]
        return {w: p for w, p in word_prob}

def prob_text(ngram_model: NGramLanguageModel, text: str, tokenizer=nltk.word_tokenize) -> float:
    """Computes the probability of the given text using the ngram language model. 
    """
    logger = logging.getLogger('prob_text')
    tokens = tokenizer(text) + [stop_symbol]
    logger.debug(f"Text tokens: {tokens}")
    p = 1
    for i in range(len(tokens)):
        p_next = ngram_model.p_next_word(tokens=tuple(tokens[:i]))
        p_token = p_next.get(tokens[i], 0)
        logger.debug(f"p({tokens[i]} | {tokens[:i]}) = {p_token} ")
        p = p*p_token
    return p


def text_generator(ngram_model: NGramLanguageModel, tokens: List[str] = None, randomize: bool = False, limit=1000):
    """Generates text using an ngram language model. 
    
    Returns a tuple (p, tokens) where p is the probability of the generated sequence of tokens, and tokens are the generated tokens. 

    The tokens can be initialized with the start of a sentence, in which case the method will complete the sentence. 

    If randomize is False, the most likely next token is picked at each step. If randomize is True, a random token is generated
    by sampling from the distribution of next tokens computed by the language model. 

    The limit parameter controls the maximum number of generated tokens. If reached, the method will halt. This is to ensure 
    that the generation loop does not enter an infinite loop, which can occur with some ngram language models. 
    
    """
    logger = logging.getLogger('text_generator')
    if tokens is None:
        tokens = []
    next = None
    p = 1
    while (next != stop_symbol):
        probs = ngram_model.p_next_word(tokens)
        if randomize:
            next = choices(list(probs.keys()), list(probs.values()), k=1)[0]
        else:
            next = list(probs.items())[0][0]  # take first item (most likely one), and take its first element (the word)
        p = p*probs[next]
        tokens.append(next)
        if len(tokens) == limit:
            logger.info(f"Reached limit of {limit} tokens!")
            break
    return (p, tokens)

def fill_in_the_gap(ngram_model: NGramLanguageModel, prefix: str, suffix: str, choices: List[str] = None, top=None):
    """Given a sentence with a gap, computes the probability distribution of filling the gap with single token.

    The sentence with the gap is passed as a prefix P and a suffix S. The method will try all concatenations of the prefix, 
    a token of the language, and the suffix, and compute the probability for each token given the prefix and suffix. 
    
    The words to be tried should be passed as the list of choices. This method is slow, we recommend limiting choices to just 
    a handful of possibilities. 
        
    """
    if choices is None: 
        choices = ngram_model.vocab
    word_prob = []
    for word in tqdm(choices):
        p_text = prob_text(ngram_model=ngram_model, text = prefix + " " + word + " " + suffix)
        word_prob.append((word, p_text))
    sum_probs = sum([wp[1] for wp in word_prob])
    word_prob = sorted(word_prob, key=lambda wp: wp[1], reverse=True)
    if top:
        word_prob = word_prob[:top]
    return {w: p/sum_probs for w, p in word_prob}
