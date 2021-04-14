from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from collections import Counter
from xml.etree import ElementTree


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def tokenize_pairs(str_):
    if str_ is None:
        return []
    return [tuple(map(int, pair.split('-'))) for pair in str_.split(' ')]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    with open(filename, 'r') as fin:
        str_ = fin.read()
    sentence_pairs, alignments = [], []
    for obj in ElementTree.fromstring(str_.replace('&', '&amp;')):
        sentence_pairs.append(SentencePair(*(obj[i].text.split(' ') for i in range(2))))
        alignments.append(LabeledAlignment(*(tokenize_pairs(obj[i].text) for i in range(2, 4))))
    return sentence_pairs, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    source_cnt, target_cnt = Counter(), Counter()
    for pair in sentence_pairs:
        source_cnt.update(pair.source)
        target_cnt.update(pair.target)
    return tuple({token: i for i, (token, _) in enumerate(cnt.most_common(freq_cutoff))}
                 for cnt in (source_cnt, target_cnt))


def tokens_to_index(tokens, vocabulary):
    inds = [vocabulary[token] for token in tokens if token in vocabulary]
    if len(inds) != len(tokens):
        return None
    return np.array(inds, dtype=np.int32)


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.

    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    pair_inds = [(tokens_to_index(pair.source, source_dict),
                  tokens_to_index(pair.target, target_dict)) for pair in sentence_pairs]
    return [TokenizedSentencePair(source, target) for source, target in pair_inds
            if source is not None and target is not None]
