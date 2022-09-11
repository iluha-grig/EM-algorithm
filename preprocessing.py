from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter


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
    sentence_pair_list = []
    labeled_alignment_list = []
    f = open(filename, 'r', encoding='utf-8')
    all_text = f.read()
    all_text = all_text.replace('&', '&amp;')
    f.close()
    parser = ET.XMLParser(encoding='utf-8')
    root = ET.fromstring(all_text, parser=parser)
    for block in root:
        if block[0].text is None:
            s1 = []
        else:
            s1 = block[0].text.split(' ')
        if block[1].text is None:
            s2 = []
        else:
            s2 = block[1].text.split(' ')
        sentence_pair_list.append(SentencePair(s1, s2))
        if block[2].text is not None:
            sure_list = block[2].text.split(' ')
        else:
            sure_list = []
        if block[3].text is not None:
            possible_list = block[3].text.split(' ')
        else:
            possible_list = []
        for i in range(len(sure_list)):
            sure_list[i] = tuple(map(int, sure_list[i].split('-')))
        for i in range(len(possible_list)):
            possible_list[i] = tuple(map(int, possible_list[i].split('-')))
        labeled_alignment_list.append(LabeledAlignment(sure_list, possible_list))
    return sentence_pair_list, labeled_alignment_list


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
    source_dict = {}
    target_dict = {}
    if freq_cutoff is None:
        source_counter = 0
        target_counter = 0
        for sp in sentence_pairs:
            for word in sp.source:
                if word not in source_dict:
                    source_dict[word] = source_counter
                    source_counter += 1
            for word in sp.target:
                if word not in target_dict:
                    target_dict[word] = target_counter
                    target_counter += 1
    else:
        counter_source = Counter()
        counter_target = Counter()
        for sp in sentence_pairs:
            counter_source += Counter(sp.source)
            counter_target += Counter(sp.target)
        most_common_source = counter_source.most_common()
        most_common_target = counter_target.most_common()
        for index, token in enumerate(most_common_source):
            if index >= freq_cutoff:
                break
            source_dict[token[0]] = index
        for index, token in enumerate(most_common_target):
            if index >= freq_cutoff:
                break
            target_dict[token[0]] = index
    return source_dict, target_dict


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
    res_list = []
    for sp in sentence_pairs:
        arr_source = np.array([], dtype=np.int32)
        arr_target = np.array([], dtype=np.int32)
        flag = False
        for word in sp.source:
            if word in source_dict:
                arr_source = np.append(arr_source, source_dict[word])
            else:
                flag = True
                break
        if flag:
            continue
        for word in sp.target:
            if word in target_dict:
                arr_target = np.append(arr_target, target_dict[word])
            else:
                flag = True
                break
        if flag:
            continue
        res_list.append(TokenizedSentencePair(arr_source, arr_target))
    return res_list


# if __name__ == '__main__':
#     print(tokenize_sents([SentencePair(['1', '2'], ['1', '3']), SentencePair(['1', '2'], ['1', '4'])],
#                          {'1': 1, '2': 2}, {'1': 1, '4': 4}))
