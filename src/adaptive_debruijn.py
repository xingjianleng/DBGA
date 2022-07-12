from collections import Counter
from debruijn import deBruijn, load_sequences, get_kmers
from itertools import chain
import math
from typing import Tuple, List, Union

from cogent3.align import make_dna_scoring_dict, global_pairwise
from cogent3.format.fasta import alignment_to_fasta


def calculate_k(seqs: Tuple[str, ...], thresh: int) -> int:
    k = math.ceil(math.log(min(map(len, seqs)), 4))
    while count_merge_node(seqs, k) > thresh:
        k *= 2
    lower = k // 2
    upper = k
    # TODO: binary search here
    for k_trial in range(lower, upper + 1):
        merge_count = count_merge_node(seqs, k_trial)
        if merge_count < thresh:
            if merge_count == 0:
                return -1
            else:
                return k_trial


def count_merge_node(seqs: Tuple[str, ...], k: int) -> int:
    kmers_seqs = []
    for seq in seqs:
        kmers_seqs.append(get_kmers(seq, k))
    # inner-sequences duplicate counts
    duplicates_within = 0
    for kmers in kmers_seqs:
        counter = Counter(kmers)
        duplicates_within += sum(value - 1 for value in counter.values())
    # inter-seuqneces duplicate counts
    counter = Counter(chain(*kmers_seqs))
    total = sum(value - 1 for value in counter.values())
    return total - duplicates_within


def dbg_alignment_recursive(data, thresh: int, s: dict, d: int, e: int) -> str:
    seqs_collection = load_sequences(data, moltype="dna")
    aln = dbg_alignment_recursive(seqs_collection.seqs, thresh, s, d, e)
    return alignment_to_fasta(
        {seqs_collection.names[0]: aln[0], seqs_collection.names[1]: aln[1]}
    )


def dbg_alignment_recursive(
    seqs: Tuple[str, ...], thresh: int, s: dict, d: int, e: int
) -> Tuple[str, ...]:
    # Base case: If the sequences are short enough, call Cogent3 alignment
    mean_len = sum(map(len, seqs)) / len(seqs)
    if mean_len < thresh:
        return global_pairwise(*seqs.seqs, s, d, e).seqs

    # 1. The initial k should lead to limited number of merge nodes
    k: int = calculate_k(seqs, thresh)
    # When it's impossible to find the appropriate k, call cogent3 alignment
    if k == -1:
        return global_pairwise(*seqs.seqs, s, d, e).seqs

    # 2. Construct de Bruijn graph with sequences and k
    d = deBruijn(seqs, k, moltype="dna")

    # 3. Extract bubbles (original bubble sequences, not kmer strings)
    # if we consider sequence is consist of [bubble, merge ... bubble ... merge, bubble]
    bubbles: List[List[Union[int, List[int]]]] = d.expansion

    aln = [], []
    # 4. Recursive case: Within each bubble, use de Bruijn graph to align the sequences
    for i, bubble in enumerate(bubbles):
        if type(bubble) == list:
            # TODO: turn bubble indicies into bubble sequences (extract bubbles)
            bubble_aln = dbg_alignment_recursive(bubble, thresh)
            for seq_idx in range(2):
                aln[seq_idx].append(bubble_aln[seq_idx])
        else:
            # TODO: extract duplicate kmers on edges
            for seq_idx in range(2):
                # Also append the merge node nucleotide
                aln[seq_idx].append(d.nodes[d.merge_node_idx[i]].kmer[0])

    return "".join(aln[0]), "".join(aln[1])
