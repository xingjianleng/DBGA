from dbga.debruijn_pairwise import DeBruijnPairwise
import math
from typing import Any, Tuple, List, Union

from cogent3.align import make_dna_scoring_dict
from cogent3.format.fasta import alignment_to_fasta
from dbga.utils import (
    dna_global_aln,
    get_closest_odd,
    get_seqs_entropy,
    load_sequences,
    NodeType,
    predict_final_p,
)


def adpt_dbg_alignment(
    data: Any,
    moltype: str,
    match: int = 10,
    transition: int = -1,
    transversion: int = -8,
    d: int = 10,
    e: int = 2,
) -> str:  # pragma: no cover
    """the function for adaptive de Bruijn alignment

    Parameters
    ----------
    data : Any
        the input sequences, could be list/tuple of str or file path
    moltype : str
        the molecular type of the sequences
    match : int, optional
        score for two matching nucleotide, by default 10
    transition : int, optional
        cost for DNA transition mutation, by default -1
    transversion : int, optional
        cost for DNA transversion mutation, by default -8
    d : int, optional
        gap open costs, by default 10
    e : int, optional
        gap extend costs, by default 2

    Returns
    -------
    str
        the fasta representation of the alignment result
    """
    # load the sequences
    seqs_collection = load_sequences(data, moltype=moltype)

    # heuristic prediction/estimation
    min_k = max(13, math.ceil(math.log(min(map(len, seqs_collection.seqs)), 4)))
    max_k = math.ceil(math.log2(min(map(len, seqs_collection.seqs)))) + 10
    k_choice = tuple(
        [
            choice
            for choice in range(
                get_closest_odd(max_k, True), get_closest_odd(min_k, False) - 1, -2
            )
        ]
    )

    # scoring dict for aligning bubbles
    s = make_dna_scoring_dict(
        match=match, transition=transition, transversion=transversion
    )

    if (
        predict_final_p(seqs_collection) < 0.80
        or get_seqs_entropy(seqs_collection) < 11
    ):
        aln = dna_global_aln(
            str(seqs_collection.seqs[0]), str(seqs_collection.seqs[1]), s, d, e
        )
    else:
        aln = adpt_dbg_alignment_recursive(
            tuple([str(seq) for seq in seqs_collection.iter_seqs()]),
            0,
            k_choice,
            s,
            d,
            e,
        )
    return alignment_to_fasta(
        {seqs_collection.names[0]: aln[0], seqs_collection.names[1]: aln[1]}
    )


def adpt_dbg_alignment_recursive(
    seqs: Tuple[str, ...],
    k_index: int,
    k_choice: Tuple[int, ...],
    s: dict,
    d: int,
    e: int,
) -> Tuple[str, ...]:
    """the recursive function for calling adaptive de Bruijn graph alignment

    Parameters
    ----------
    seqs : Tuple[str, ...]
        the input sequences
    k_index : int
        the index for chosen k in the k_choice list
    k_choice : Tuple[int, ...]
        the possible choices for kmer size
    s : dict
        the DNA scoring dictionary
    d : int
        gap open costs
    e : int
        gap extend costs

    Returns
    -------
    Tuple[str, ...]
        the aligned sequences
    """

    # Two versions for adaptive de Bruijn alignment
    # 1. Use brute-force, descend from large k to smaller k
    # 2. Use mathematics and statistics ways to analyse the similarity between sequences to choose k
    if k_index < 0 or k_index > len(k_choice) - 1:
        return dna_global_aln(*seqs, s, d, e)

    k = k_choice[k_index]
    # Base case: When it's impossible to find the appropriate k, call cogent3 alignment
    if k <= 0 or k > min(map(len, seqs)):
        return dna_global_aln(*seqs, s, d, e)

    # 2. Construct de Bruijn graph with sequences and k
    dbg = DeBruijnPairwise(seqs, k, moltype="dna")

    # Edge case: when there's no merge node in the de Bruijn graph, directly align
    # when there's only merge node in the graph, it might lead to infinite loop, so directly align
    if len(dbg.merge_node_idx) < 2:
        return adpt_dbg_alignment_recursive(seqs, k_index + 1, k_choice, s, d, e)

    # 3. Extract bubbles (original bubble sequences, not kmer strings)
    # if we consider sequence is consist of [bubble, merge ... bubble ... merge, bubble]
    bubbles: List[List[Union[int, List[int]]]] = dbg.expansion
    expansion1, expansion2 = bubbles[0].copy(), bubbles[1].copy()

    aln = [], []
    # 4. Recursive case: Within each bubble, use de Bruijn graph to align the sequences
    # Iterate until there are the last merge node with last bubble in the expansion
    for i in range(len(expansion1) - 2):
        bubble1 = expansion1[i]
        bubble2 = expansion2[i]
        if type(bubble1) == list:

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

            bubble_aln = adpt_dbg_alignment_recursive(
                extracted_bubble, k_index + 1, k_choice, s, d, e
            )
            for seq_idx in range(2):
                aln[seq_idx].append(bubble_aln[seq_idx])
        else:
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
                        bubble_read_seq1.append(edge.duplicate_str)
                    else:
                        # if the next node isn't the end node, only add the single nucleotide
                        bubble_read_seq1.append(edge.duplicate_str[0])
            bubble_read_seq1 = "".join(bubble_read_seq1)

            # for sequence 2
            bubble_read_seq2 = [dbg.read_from_kmer(bubble2, 1)]
            if 1 in dbg.nodes[bubble2].out_edges:
                # if exist an out edge
                edge = dbg.nodes[bubble2].out_edges[1]
                if edge.duplicate_str:
                    # if the the edge contains duplicate kmers
                    if dbg.nodes[edge.in_node].node_type == NodeType.end:
                        # if the next node is the end node, add the full edge kmer
                        bubble_read_seq2.append(edge.duplicate_str)
                    else:
                        # if the next node isn't the end node, only add the single nucleotide
                        bubble_read_seq2.append(edge.duplicate_str[0])
            bubble_read_seq2 = "".join(bubble_read_seq2)

            # add the bubble read of the merge nodes into alignments
            aln[0].append(bubble_read_seq1)
            aln[1].append(bubble_read_seq2)

    # The last set of bubbles should be added with the merge node in front of it
    # in case where the last merge node is the end of any sequences
    # Should contain [last merge node, last bubble]
    tail_bubble1 = [expansion1[len(expansion1) - 2]] + expansion1[len(expansion1) - 1]
    tail_bubble2 = [expansion2[len(expansion2) - 2]] + expansion2[len(expansion2) - 1]
    extracted_bubble: Tuple[str, str] = dbg.extract_bubble_seq(
        tail_bubble1, 0
    ), dbg.extract_bubble_seq(tail_bubble2, 1)
    tail_bubble_aln = adpt_dbg_alignment_recursive(
        extracted_bubble, k_index + 1, k_choice, s, d, e
    )
    for seq_idx in range(2):
        aln[seq_idx].append(tail_bubble_aln[seq_idx])

    return "".join(aln[0]), "".join(aln[1])
