from collections import Counter
from re import L
from debruijn import deBruijn, load_sequences, get_kmers, NodeType, dna_global_aln
from itertools import chain
import math
from typing import Tuple, List, Union

from cogent3.align import make_dna_scoring_dict, global_pairwise
from cogent3.format.fasta import alignment_to_fasta


def calculate_k(seqs: Tuple[str, ...], thresh: int) -> int:
    """the function for calculating an appropriate k for minimum number of merge nodes

    Parameters
    ----------
    seqs : Tuple[str, ...]
        the input collection of sequences
    thresh : int
        the threshold for the miminum number of merge nodes in the de Bruijn graph

    Returns
    -------
    int
        the suggested k for the de Bruijn graph
    """

    k = math.ceil(math.log(min(map(len, seqs)), 4))
    while count_merge_node(seqs, k) > thresh or k > min(map(len, seqs)):
        k += 1
    return k


def count_merge_node(seqs: Tuple[str, ...], k: int) -> int:
    kmers_seqs = []
    for seq in seqs:
        kmers_seqs.append(get_kmers(seq, k))
    # inner-sequences duplicate counts
    duplicates_kmer_set = set()
    for kmers in kmers_seqs:
        counter = Counter(kmers)
        for kmer, count in counter.items():
            if count > 1:
                duplicates_kmer_set.add(kmer)
    # inter-seuqneces duplicate counts
    counter = Counter(chain(*kmers_seqs))
    total = 0
    for kmer, count in counter.items():
        if count > 1 and kmer not in duplicates_kmer_set:
            total += 1
    return total


def dbg_alignment(data, thresh: int, s: dict, d: int, e: int) -> str:
    seqs_collection = load_sequences(data, moltype="dna")
    aln = dbg_alignment_recursive(
        tuple([str(x) for x in seqs_collection.iter_seqs()]), thresh, s, d, e
    )
    return alignment_to_fasta(
        {seqs_collection.names[0]: aln[0], seqs_collection.names[1]: aln[1]}
    )


def dbg_alignment_recursive(
    seqs: Tuple[str, ...], thresh: int, s: dict, d: int, e: int
) -> Tuple[str, ...]:
    # Base case: If the sequences are short enough, call Cogent3 alignment
    min_len = min(map(len, seqs))
    if min_len < thresh:
        return dna_global_aln(*seqs, s, d, e)

    # 1. The initial k should lead to limited number of merge nodes
    k: int = calculate_k(seqs, thresh)
    # When it's impossible to find the appropriate k, call cogent3 alignment
    if k < 0 or k > min(map(len, seqs)):
        return dna_global_aln(*seqs, s, d, e)

    # 2. Construct de Bruijn graph with sequences and k
    dbg = deBruijn(seqs, k, moltype="dna")

    # FIXME: Edge case: when there's no merge node in the de Bruijn graph, directly align
    if not dbg.merge_node_idx:
        return dna_global_aln(*seqs, s, d, e)

    # 3. Extract bubbles (original bubble sequences, not kmer strings)
    # if we consider sequence is consist of [bubble, merge ... bubble ... merge, bubble]
    bubbles: List[List[Union[int, List[int]]]] = dbg.expansion
    expansion1, expansion2 = bubbles[0], bubbles[1]

    aln = [], []
    # 4. Recursive case: Within each bubble, use de Bruijn graph to align the sequences
    # FIXME: The last set of bubbles should be added with the merge node in front of it
    # in case where the last merge node is the end of the sequence
    for i in range(len(expansion1)):
        if type(expansion1[i]) == list:
            bubble1 = expansion1[i].copy()
            bubble2 = expansion2[i].copy()

            # NOTE: a list must be followed by an integer
            # however, a integer might be followed by another integer
            # turn bubble indicies into bubble sequences (extract bubbles)
            # read extract_bubble_seq function from the deBruijn class
            if i != len(expansion1) - 1:
                # if bubble is not the last element, it must be followed by a merge node
                bubble1.append(expansion1[i + 1])
                bubble2.append(expansion2[i + 1])
            extracted_bubble: Tuple[str, str] = dbg.extract_bubble_seq(
                bubble1, 0
            ), dbg.extract_bubble_seq(bubble2, 1)

            bubble_aln = dbg_alignment_recursive(extracted_bubble, thresh, s, d, e)
            for seq_idx in range(2):
                aln[seq_idx].append(bubble_aln[seq_idx])
        else:
            bubble1 = expansion1[i]
            bubble2 = expansion2[i]
            # extract duplicate kmers on edges
            # read the common part of the node (need to determine whether it's the end node)

            # for sequence 1, need two separate cases becasue it might end with separated branches
            # merge node for seq 1
            bubble_read_seq1 = [dbg.read_from_kmer(bubble1, 0)]
            if 0 in dbg.nodes[bubble1].out_edges:
                # if exist an out edge
                edge = dbg.nodes[bubble1].out_edges[0]
                if edge.duplicate_str:
                    # if the the edge contains duplicate kmers
                    if dbg.nodes[edge.in_node].node_type == NodeType.end:
                        # if the next node is the end node, add the full edge kmer
                        bubble_read_seq1.append(edge.kmer)
                    else:
                        # if the next node isn't the end node, only add the single nucleotide
                        bubble_read_seq1.append(edge.kmer[0])
            bubble_read_seq1 = "".join(bubble_read_seq1)
            # print(bubble_read_seq1)

            # for sequence 2
            bubble_read_seq2 = [dbg.read_from_kmer(bubble2, 1)]
            if 1 in dbg.nodes[bubble2].out_edges:
                # if exist an out edge
                edge = dbg.nodes[bubble2].out_edges[1]
                if edge.duplicate_str:
                    # if the the edge contains duplicate kmers
                    if dbg.nodes[edge.in_node].node_type == NodeType.end:
                        # if the next node is the end node, add the full edge kmer
                        bubble_read_seq2.append(edge.kmer)
                    else:
                        # if the next node isn't the end node, only add the single nucleotide
                        bubble_read_seq2.append(edge.kmer[0])
            bubble_read_seq2 = "".join(bubble_read_seq2)
            # print(bubble_read_seq2)

            # add the bubble read of the merge nodes into alignments
            aln[0].append(bubble_read_seq1)
            aln[1].append(bubble_read_seq2)

    return "".join(aln[0]), "".join(aln[1])
