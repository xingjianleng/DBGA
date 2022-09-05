from __future__ import annotations
from collections import Counter
from enum import Enum, auto
from itertools import combinations
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Union

from cogent3.align import global_pairwise
from cogent3.align.progressive import TreeAlign
from cogent3.evolve.fast_distance import DistanceMatrix
from cogent3 import load_unaligned_seqs, make_unaligned_seqs
from cogent3 import SequenceCollection
import graphviz


def list_intersection(list1: List[Any], list2: List[Any]) -> List[Any]:
    """calculate the intersection of two lists with runtime O(nlogn)

    Parameters
    ----------
    list1 : List[Any]
        the first input list
    list2 : List[Any]
        the second input list

    Returns
    -------
    List[Any]
        the intersection of two input lists
    """
    rtn = []
    # for merge nodes, sort the list
    lst1_sorted = sorted(list1)
    lst2_sorted = sorted(list2)
    ptr1, ptr2 = 0, 0
    while ptr1 < len(lst1_sorted) and ptr2 < len(lst2_sorted):
        if lst1_sorted[ptr1] == lst2_sorted[ptr2]:
            rtn.append(lst1_sorted[ptr1])
            ptr1 += 1
            ptr2 += 1
        elif lst1_sorted[ptr1] < lst2_sorted[ptr2]:
            ptr1 += 1
        else:
            ptr2 += 1
    # no intersection if one list run out of elements
    return rtn


def read_nucleotide_from_kmers(seq: str, k: int) -> str:
    """Read the kmer(s) contained in edges in a de Bruijn graph

    Parameters
    ----------
    seq : str
        the duplicate string sequence from edges of de Bruijn graph
    k : int
        the kmer size of de Bruijn graph

    Returns
    -------
    str
        the edge kmer read

    """
    assert len(seq) % k == 0
    rtn = [seq[i] for i in range(0, len(seq), k)]
    return "".join(rtn)


def load_sequences(data: Any, moltype: str) -> SequenceCollection:
    """Load the sequences and transform it into numpy array of strings (in Unicode)

    Parameters
    ----------
    data : Any
        sequences to load, could be `path`, `SequenceCollection`, `list`
    moltype : str
        the molecular type in the sequence

    Returns
    -------
    SequenceCollection
        the SequenceCollection object of the loaded sequences

    Raises
    ------
    ValueError
        if the sequences parameter is not `path`, `SequenceCollection`, `list`, `tuple`

    """

    if isinstance(data, str) or isinstance(data, Path):
        data = load_unaligned_seqs(data, moltype=moltype)
    elif (isinstance(data, list) or isinstance(data, tuple)) and all(
        isinstance(elem, str) for elem in data
    ):
        dict_data = {index: seq for index, seq in enumerate(data)}
        data = make_unaligned_seqs(dict_data, moltype=moltype)
    if isinstance(data, SequenceCollection):
        return data
    else:
        raise ValueError("Invalid input for sequence argument")


def get_kmers(sequence: str, k: int) -> List[str]:
    """Get the kmers in sequences

    Parameters
    ----------
    sequence : str
        the sequence for calculating kmers
    k : int
        k size for each kmer

    Returns
    -------
    List[str]
        the list of kmers of the sequence

    Raises
    ------
    ValueError
        the k value should be in [1, len(sequence)]

    """
    if k < 0 or k > len(sequence):
        raise ValueError("Invalid k size for kmers")
    return [sequence[i : i + k] for i in range(len(sequence) - k + 1)]


def duplicate_kmers(kmer_seqs: List[List[str]]) -> Set[str]:
    """Get the duplicate kmers from each sequence

    Parameters
    ----------
    kmer_seqs : List[List[str]]
        list of list of kmers for each sequence

    Returns
    -------
    Set[str]
        the set containing duplicate kmers

    """
    duplicate_set = set()
    for kmer_seq in kmer_seqs:
        counter = Counter(kmer_seq)
        for key, value in counter.items():
            if value > 1:
                duplicate_set.add(key)
    return duplicate_set


def dna_global_aln(
    seq1: str, seq2: str, s: Dict[Tuple[str, str], int], d: int, e: int
) -> Tuple[str, str]:
    """Align the sequences in bubbles with node indices provided

    Parameters
    ----------
    seq1 : str
        the first sequence to align
    seq2 : str
        the second sequence to align
    moltype : str
        the molecular type in the sequence
    s : Dict[Tuple[str, str], int]
        the DNA scoring matrix
    d : int
        gap open costs
    e : int
        gap extend costs

    Returns
    -------
    Tuple[str, str]
        the tuple of aligned sequences

    """
    if seq1 and seq2:
        seq_colllection = make_unaligned_seqs({0: seq1, 1: seq2}, moltype="dna")
        partial_aln = global_pairwise(*seq_colllection.seqs, s, d, e)
        return str(partial_aln.seqs[0]), str(partial_aln.seqs[1])
    elif seq1:
        return seq1, "-" * len(seq1)
    elif seq2:
        return "-" * len(seq2), seq2
    else:
        return "", ""


def dna_msa(
    seqs: SequenceCollection,
    model: str = "F81",
    tree: Any = None,
    indel_rate: float = 0.01,
    indel_length: float = 0.01,
) -> SequenceCollection:
    """_summary_

    Parameters
    ----------
    seqs : SequenceCollection
        the SequenceCollection object of all sequences that need to be aligned
    model : str, optional
        a substitution model or the name of one, see cogent3.available_models(), by default "F81"
    tree : Any, optinal
        a guide tree for multiple sequence alignment, by default None
    indel_rate : float, optional
        one parameter for the progressive pair-HMM, by default 0.01
    indel_length : float, optional
        one parameter for the progressive pair-HMM, by default 0.01

    Returns
    -------
    SequenceCollection
        the SequenceCollection representation of the multiple-sequence alignment result
    """
    seqs_not_non = seqs.take_seqs_if(lambda seq: len(seq) > 0)
    seqs_non = seqs.take_seqs_if(lambda seq: len(seq) > 0, negate=True)
    if type(seqs_non) == dict:
        # all sequences not null
        aln, _ = TreeAlign(
            model=model,
            seqs=seqs_not_non,
            tree=tree,
            indel_rate=indel_rate,
            indel_length=indel_length,
        )
        return aln
    elif type(seqs_not_non) == dict:
        # case where all sequences is empty
        return seqs
    elif seqs_not_non.num_seqs == 1:
        # case where only one sequence is not empty
        return seqs_not_non.add_seqs(seqs_non)
    else:
        aln, _ = TreeAlign(
            model=model,
            seqs=seqs_not_non,
            tree=tree,
            indel_rate=indel_rate,
            indel_length=indel_length,
        )
        seqs_non_sc = make_unaligned_seqs(
            {name: aln.seq_len * "-" for name in seqs_non.names}, moltype=seqs.moltype
        )
        return aln.add_seqs(seqs_non_sc)


def debruijn_merge_correctness(
    seqs_expansion: List[List[Union[int, List[int]]]]
) -> bool:
    """determine whether de Bruijn graph contains ambiguous cycles

    Parameters
    ----------
    seqs_expansion : List[List[Union[int, List[int]]]]
        the expansion of sequences into [bubble, merge ... merge, bubble] form

    Returns
    -------
    bool
        whether de Bruijn graph contains ambiguous cycles
    """
    if not all(elem == len(seqs_expansion[0]) for elem in map(len, seqs_expansion)):
        return False
    else:
        for i in range(len(seqs_expansion[0])):
            if type(seqs_expansion[0][i]) == type(seqs_expansion[1][i]) == int:
                if not seqs_expansion[0][i] == seqs_expansion[1][i]:
                    return False
            else:
                if not type(seqs_expansion[0][i]) == type(seqs_expansion[1][i]) == list:
                    return False
        return True


def to_DOT(nodes: List[Node]) -> graphviz.Digraph:
    """Obtain the DOT representation to the de Bruijn graph

    Parameters
    ----------
    nodes : List[Node]
        list of nodes that belong to a de Bruijn graph

    Returns
    -------
    graphviz.Digraph
        graphviz Digraph object representing the de Bruijn graph

    """
    dot = graphviz.Digraph("de Bruijn graph")
    for node in nodes:
        dot.node(str(node.id), f"{str(node.id)}.{node.kmer}")
    for node in nodes:
        for other_node, edge in node.next_nodes.items():
            dot.edge(
                tail_name=str(node.id),
                head_name=str(other_node),
                label=edge.duplicate_str,
                weight=str(edge.multiplicity),
            )
    return dot


def predict_p(seqs: SequenceCollection, k: int) -> float:
    """predict similarity for of two sequences with the k size given using IoU of kmers

    Parameters
    ----------
    seqs : SequenceCollection
        SequenceCollection object containing two sequences
    k : int
        kmer size chosen

    Returns
    -------
    float
        the predicted similarity of sequences for given k size
    """
    # seqs is a SequenceCollection object
    kmers_seq0_set = set(get_kmers(str(seqs.seqs[0]), k))
    kmers_seq1_set = set(get_kmers(str(seqs.seqs[1]), k))
    set_union = kmers_seq0_set | kmers_seq1_set
    set_intersection = kmers_seq0_set & kmers_seq1_set

    p_prime = len(set_intersection) / len(set_union)
    p = p_prime ** (1 / k)
    return p


def predict_final_p(seqs: SequenceCollection) -> float:
    """estimate the general similarity between two sequences using IoU of kmers

    Parameters
    ----------
    seqs : SequenceCollection
        the SequenceCollection containing pairwise sequences

    Returns
    -------
    float
        the predicted general similarity of sequences for given k size
    """
    min_k = math.ceil(math.log(min(map(len, seqs.seqs)), 4))
    max_k = math.ceil(math.log2(min(map(len, seqs.seqs)))) + 10
    return min([p for p in (predict_p(seqs, k) for k in range(min_k, max_k))])


def tree_prediction(seq_sc: SequenceCollection) -> Any:
    """generate the estimated tree for input sequences using the predicted pairwise similarity

    Parameters
    ----------
    seq_sc : SequenceCollection
        the SequenceCollection containing multiple sequences

    Returns
    -------
    Any
        the estimated tree using the predicted pairwise similarity
    """
    dist_dict = {}
    for seq_name1, seq_name2 in combinations(seq_sc.names, r=2):
        sub_seqs = seq_sc.take_seqs((seq_name1, seq_name2))
        distance = 1 - predict_final_p(sub_seqs)
        dist_dict[(seq_name1, seq_name2)] = distance
        dist_dict[(seq_name2, seq_name1)] = distance
    dm = DistanceMatrix(dist_dict)
    tree = dm.quick_tree(True)
    return tree


# NodeType class to indicate the type of node (start/middle/end)
class NodeType(Enum):
    start = auto()
    middle = auto()
    end = auto()


class Edge:
    """The Edge class to connect two nodes

    Attributes
    ----------
    out_node : int
        the node_id of the starting node
    in_node : int
        the node_id of the terminal node
    duplicate_str : str, optional
        the edge can represent duplicate kmers

    """

    def __init__(
        self, out_node: int, in_node: int, duplicate_str: str, multiple_duplicate: bool
    ) -> None:
        """Constructor for the Edge class

        Parameters
        ----------
        out_node : int
            the node_id of the starting node
        in_node : int
            the node_id of the terminal node
        duplicate_str : str
            the edge can represent duplicate kmers. Defaults to ""
        multiple_duplicate : bool
            indicate whether the duplicate_str contain multiple kmers

        Returns
        -------

        """
        self.out_node = out_node
        self.in_node = in_node
        self.duplicate_str = duplicate_str
        self.multiple_duplicate = multiple_duplicate
        self.multiplicity = 1


class Node:
    """The Node class to construct a de Bruijn graph

    Attributes
    ----------
    id : int
        the id of the node
    kmer : str
        the kmer string of the node
    node_type : NodeType
        the NodeType of the node (start/middle/end)

    """

    def __init__(self, id: int, kmer: str, node_type: NodeType) -> None:
        """Constructor for Node class

        Parameters
        ----------
        id : int:
            the id of the node
        kmer : str
            the kmer string of the node
        node_type : NodeType
            the NodeType of the node (start/middle/end)

        Returns
        -------

        """
        self.id = id
        self.kmer = kmer
        self.node_type = node_type
        # the edge between two nodes won't be duplicate in the same direction
        # out_edges maps from seq_idx to the corresponding out edge
        self.out_edges: Dict[int, Edge] = {}
        # use the dictionary to map from next node index to the Edge object
        self.next_nodes: Dict[int, Edge] = {}
        self.count = 1

    def add_out_edge(
        self, other_id: int, seq_idx: int, duplicate_str: str, multiple_duplicate: bool
    ) -> None:
        """Add the edge from this node to other node

        Parameters
        ----------
        other_id : int
            the terminal node id of the edge
        seq_idx : int
            the sequence index of the current kmer connection
        duplicated_str : str
            the duplicated kmer represented by the edge. Defaults to ""
        multiple_duplicate : bool
            indicate whether the duplicate_str contain multiple kmers

        Returns
        -------

        """
        if seq_idx in self.out_edges:
            self.out_edges[seq_idx].multiplicity += 1
        else:
            new_edge = Edge(self.id, seq_idx, duplicate_str, multiple_duplicate)
            self.out_edges[seq_idx] = new_edge
            self.next_nodes[other_id] = new_edge
