from typing import Tuple, List, Union
from debruijn import deBruijn, load_sequences

from cogent3.align import make_dna_scoring_dict, global_pairwise
from cogent3.format.fasta import alignment_to_fasta


def calculate_k(seqs: Tuple[str, ...]) -> int:
    pass


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
    k: int = calculate_k(seqs)
    # When it's impossible to find the appropriate k, call cogent3 alignment
    if k == -1:
        return global_pairwise(*seqs.seqs, s, d, e).seqs

    # 2. Construct de Bruijn graph with sequences and k
    d = deBruijn(seqs, k, moltype="dna")

    # 3. Extract bubbles (original bubble sequences, not kmer strings)
    # if we consider sequence is consist of [bubble, merge ... bubble ... merge, bubble]
    bubbles: List[List[Union[int, List[int]]]] = d.extract_bubble_indicies()

    # The condition should be satisfied
    assert len(d.merge_node_idx) == len(bubbles) - 1

    aln = [], []
    # 4. Recursive case: Within each bubble, use de Bruijn graph to align the sequences
    for i, bubble in enumerate(bubbles):
        bubble_aln = dbg_alignment_recursive(bubble, thresh)
        for seq_idx in range(2):
            aln[seq_idx].append(bubble_aln[seq_idx])
        if i != len(bubbles) - 1:
            for seq_idx in range(2):
                # Also append the merge node nucleotide
                aln[seq_idx].append(d.nodes[d.merge_node_idx[i]].kmer[0])

    return "".join(aln[0]), "".join(aln[1])
