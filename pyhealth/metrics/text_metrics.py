# Author: Jingtong Xing (jxing11)
# Email: jxing11@illinois.edu
# Paper: Based on concepts learned from reproducing Uncertainty-Aware Text-to-Program for QA on Structured EHR (Kim et al., CHIL 2022)
# Link: https://arxiv.org/abs/2203.06918
# Description: Implements Program Inconsistency Score (PIS) metric using Levenshtein distance.

import Levenshtein
from typing import List
import numpy as np

def calculate_program_inconsistency_score(program_list: List[str]) -> float:
    """
    Calculates the Program Inconsistency Score (PIS) for a list of strings.

    PIS is defined as the average pairwise normalized Levenshtein distance
    between all unique pairs of items in the list. A higher score indicates
    more inconsistency (diversity) among the items. This can be useful for
    evaluating the diversity of outputs from generative models, for example.

    The Levenshtein distance is normalized by the length of the longer string
    in each pair, yielding a score between 0 (identical) and 1 (completely different)
    for each pair. The PIS is the average of these normalized distances.

    Args:
        program_list: A list of strings.

    Returns:
        A float representing the PIS, ranging from 0.0 (all items identical
        or fewer than two items) to 1.0 (all items maximally different, on average).
    """
    if not program_list or len(program_list) < 2:
        return 0.0

    total_normalized_distance = 0.0
    num_comparisons = 0

    for i in range(len(program_list)):
        for j in range(i + 1, len(program_list)):
            prog1 = program_list[i]
            prog2 = program_list[j]

            # Levenshtein distance
            dist = Levenshtein.distance(prog1, prog2)

            # Normalized by length of longer string
            max_len = max(len(prog1), len(prog2))
            # handle case where both are empty strings
            if max_len == 0:  
                normalized_dist = 0.0
            else:
                normalized_dist = dist / float(max_len)

            total_normalized_distance += normalized_dist
            num_comparisons += 1

    # to avoid division by zero
    return total_normalized_distance / float(num_comparisons) if num_comparisons > 0 else 0.0 