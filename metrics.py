from typing import List, Tuple

from preprocessing import LabeledAlignment


def compute_precision(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the precision for predicted alignments.
    Numerator : |predicted and possible|
    Denominator: |predicted|
    Note that for correct metric values `sure` needs to be a subset of `possible`, but it is not the case for input data.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and possible sets, summed over all sentences
        total_predicted: total number of predicted alignments over all sentences
    """
    numerator = 0
    denominator = 0
    for la, predicted_al in zip(reference, predicted):
        set_sure = set(la.sure)
        set_possible = set(la.possible)
        set_predicted = set(predicted_al)
        set_possible |= set_sure
        denominator += len(set_predicted)
        numerator += len(set_possible & set_predicted)
    return numerator, denominator


def compute_recall(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the recall for predicted alignments.
    Numerator : |predicted and sure|
    Denominator: |sure|

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and sure sets, summed over all sentences
        total_predicted: total number of sure alignments over all sentences
    """
    numerator = 0
    denominator = 0
    for la, predicted_al in zip(reference, predicted):
        set_sure = set(la.sure)
        set_predicted = set(predicted_al)
        denominator += len(set_sure)
        numerator += len(set_predicted & set_sure)
    return numerator, denominator


def compute_aer(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> float:
    """
    Computes the alignment error rate for predictions.
    AER=1-(|predicted and possible|+|predicted and sure|)/(|predicted|+|sure|)
    Please use compute_precision and compute_recall to reduce code duplication.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        aer: the alignment error rate
    """
    pr_num, pr_den = compute_precision(reference, predicted)
    re_num, re_den = compute_recall(reference, predicted)
    return 1 - (pr_num + re_num) / (pr_den + re_den)
