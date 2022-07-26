from collections import Counter
from debruijn import deBruijn, load_sequences, get_kmers, NodeType, dna_global_aln
from itertools import chain
import math
from pathlib import Path
from typing import Any, Tuple, List, Union

import click
from cogent3.align import make_dna_scoring_dict
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
    """estimate the number of merge nodes in the de Bruijn graph for the given kmer size

    Parameters
    ----------
    seqs : Tuple[str, ...]
        the input sequences
    k : int
        the given kmer size

    Returns
    -------
    int
        the estimated number of merge nodes
    """
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


def adpt_dbg_alignment(
    data: Any,
    thresh: int,
    moltype: str,
    match: int = 10,
    transition: int = -1,
    transversion: int = -8,
    d: int = 10,
    e: int = 2,
) -> str:
    """the function for adaptive de Bruijn alignment

    Parameters
    ----------
    data : Any
        the input sequences, could be list/tuple of str or file path
    thresh : int
        the maximum number of merge nodes allowed in the de Bruijn graph
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

    # scoring dict for aligning bubbles
    s = make_dna_scoring_dict(
        match=match, transition=transition, transversion=transversion
    )
    aln = adpt_dbg_alignment_recursive(
        tuple([str(x) for x in seqs_collection.iter_seqs()]), thresh, s, d, e
    )
    return alignment_to_fasta(
        {seqs_collection.names[0]: aln[0], seqs_collection.names[1]: aln[1]}
    )


def adpt_dbg_alignment_recursive(
    seqs: Tuple[str, ...], thresh: int, s: dict, d: int, e: int
) -> Tuple[str, ...]:
    """the recursive function for calling adaptive de Bruijn graph alignment

    Parameters
    ----------
    seqs : Tuple[str, ...]
        the input sequences
    thresh : int
        the maximum number of merge nodes allowed in the de Bruijn graph
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

    # Edge case: when there's no merge node in the de Bruijn graph, directly align
    # when there's only merge node in the graph, it might lead to infinite loop, so directly align
    if len(dbg.merge_node_idx) < 2:
        return dna_global_aln(*seqs, s, d, e)

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

            bubble_aln = adpt_dbg_alignment_recursive(extracted_bubble, thresh, s, d, e)
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
                        bubble_read_seq2.append(edge.duplicate_str)
                    else:
                        # if the next node isn't the end node, only add the single nucleotide
                        bubble_read_seq2.append(edge.duplicate_str)
            bubble_read_seq2 = "".join(bubble_read_seq2)
            # print(bubble_read_seq2)

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
    tail_bubble_aln = adpt_dbg_alignment_recursive(extracted_bubble, thresh, s, d, e)
    for seq_idx in range(2):
        aln[seq_idx].append(tail_bubble_aln[seq_idx])

    return "".join(aln[0]), "".join(aln[1])


@click.command()
@click.option(
    "--infile", type=str, required=True, help="input unaligned sequences file"
)
@click.option(
    "--outfile", type=str, required=True, help="output aligned file destination"
)
@click.option(
    "--thresh",
    type=int,
    required=True,
    help="threshold for maximum number of merge nodes",
)
@click.option(
    "--moltype",
    default="dna",
    type=str,
    required=False,
    help="molecular type of sequences",
)
@click.option(
    "--match",
    default=10,
    type=int,
    required=False,
    help="score for two matching nucleotide",
)
@click.option(
    "--transition",
    default=-1,
    type=int,
    required=False,
    help="cost for DNA transition mutation",
)
@click.option(
    "--transversion",
    default=-8,
    type=int,
    required=False,
    help="cost for DNA transversion mutation",
)
@click.option(
    "--d", default=10, type=int, required=False, help="costs for opening a gap"
)
@click.option(
    "--e", default=2, type=int, required=False, help="costs for extending a gap"
)
def cli(infile, outfile, thresh, moltype, match, transition, transversion, d, e):
    aln = adpt_dbg_alignment(
        infile, thresh, moltype, match, transition, transversion, d, e
    )
    out_path = Path(outfile)
    with open(f"{out_path.stem}_thresh{thresh}{out_path.suffix}", "w") as f:
        f.write(aln)


if __name__ == "__main__":
    cli()
