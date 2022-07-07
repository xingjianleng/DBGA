from __future__ import annotations
from collections import Counter
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Union

import click
from cogent3.align import global_pairwise, make_dna_scoring_dict
from cogent3.format.fasta import alignment_to_fasta
from cogent3 import load_unaligned_seqs, make_unaligned_seqs
from cogent3 import SequenceCollection
import graphviz
import numpy as np
import plotly.express as px


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
        if the sequences parameter is not `path`, `SequenceCollection`, `list`

    """

    if isinstance(data, str):
        path = Path(data).expanduser().absolute()
        data = load_unaligned_seqs(path, moltype=moltype)
    elif isinstance(data, list) and all(isinstance(elem, str) for elem in data):
        data = make_unaligned_seqs(data, moltype=moltype)
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
    assert len(kmer_seqs) == 2
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


def to_DOT(nodes: List[Node]) -> graphviz.Digraph:  # pragma: no cover
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
    dot = graphviz.Digraph("de Bruijn")
    for node in nodes:
        dot.node(str(node.id), node.kmer)
    for node in nodes:
        for other_node, edge in node.out_edges.items():
            dot.edge(
                tail_name=str(node.id),
                head_name=str(other_node),
                label=edge.duplicate_str,
                weight=str(edge.multiplicity),
            )
    return dot


def chebyshev(arr: np.ndarray, coef: float) -> np.ndarray:  # pragma: no cover
    """Use Chebyshev to calculate outliers

    Parameters
    ----------
    arr : np.ndarray
        the array of data points
    coef : float
        the coefficient for how many standard deviations to be considered as an outlier

    Returns
    -------
    np.ndarray
        the ndarray of booleans indicating whether the element is an outlier
    """
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    cheb = coef * std
    outliers = np.logical_or(arr < (mean - cheb), arr > (mean + cheb))
    assert outliers.shape == arr.shape
    return outliers


def filter_outliers(
    indices: np.ndarray, arr: np.ndarray, window: int, thresh: float, coef: float
) -> np.ndarray:  # pragma: no cover
    """Filter outliers in a given array

    Parameters
    ----------
    indices : np.ndarray
        the indices of each array element
    arr : np.ndarray
        the array elements
    window : int
        the window size for local Chebyshev algorithm
    thresh : float
        the threshold for a data point to be considered as an outlier
    coef : float
        the coefficient for how many standard deviations to be considered as an outlier

    Returns
    -------
    np.ndarray
        the list of indices after filtering

    """
    # use local Chebyshev to filter remaining outliers
    assert indices.shape == arr.shape
    window_num = indices.size - window + 1
    outlier_count = np.zeros_like(indices)
    for i in range(window_num):
        outlier_count[i : i + window] += chebyshev(arr[i : i + window], coef)

    inliers = outlier_count <= window * thresh
    return indices[inliers]


def merge_indices_fix(
    db: deBruijn, thresh: float = 0.1
) -> List[int]:  # pragma: no cover
    """Fix function when cycle in merge node indices list can be observed

    Parameters
    ----------
    db : deBruijn
        a deBruijn graph object of two sequences
    thresh : float, optional
        the threshold for a data point to be considered as an outlier. Defaults to 0.1

    Returns
    -------
    List[int]
        the fixed merge node indices list

    """
    merge_shifts = mapping_shifts(db)
    indices_arr = np.array(merge_shifts["node_indices"])
    shifts_arr = np.array(merge_shifts["shifts"])
    coef = min(
        3, max(0.3, (db.avg_len + 4.4) / 18)
    )  # heuristic to calculate coefficient
    if db.avg_len <= 300:  # heuristic to calculate window size
        window = max(5, round(db.avg_len / 15))
    elif db.avg_len <= 1000:
        window = max(20, round(db.avg_len / 25))
    else:
        window = max(40, round(db.avg_len / 600))
    print(window, thresh, coef)
    filtered_indices = filter_outliers(
        indices_arr, shifts_arr, window=window, thresh=thresh, coef=coef
    )
    return filtered_indices.tolist()


def mapping_shifts(db: deBruijn, visualize: bool = False) -> Dict[str, List[int]]:
    """The function to plot the shifting of indices of matched kmers

    Parameters
    ----------
    db : deBruijn
        a deBruijn graph object of two sequences
    visualize : bool, optional
        whether show the visualization of the shifts plot. Defaults to False

    Returns
    -------
    Dict[List[int], List[int]]
        the dictionary containing merge_node_indices and the shift values

    """
    # db should be in type deBruijn
    assert isinstance(db, deBruijn)

    sorted_merge: List[int] = sorted(db.merge_node_idx)

    # TODO: Need a more efficient way to calculate/memorize the indices
    shifts = [
        db.seq_node_idx[1].index(merge_idx) - db.seq_node_idx[0].index(merge_idx)
        for merge_idx in sorted_merge
    ]

    merge_shifts = {
        "node_indices": sorted_merge,
        "shifts": shifts,
    }

    if visualize:  # pragma: no cover
        figure = px.line(
            merge_shifts,
            x="node_indices",
            y="shifts",
            title="Matching shifts over merge node indices based on sequence 1",
            markers=True,
        )
        figure.show()

    return merge_shifts


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

    def __init__(self, out_node: int, in_node: int, duplicate_str: str = "") -> None:
        """Constructor for the Edge class

        Parameters
        ----------
        out_node : int
            the node_id of the starting node
        in_node : int
            the node_id of the terminal node
        duplicate_str : str, optional
            the edge can represent duplicate kmers. Defaults to ""

        Returns
        -------

        """
        self.out_node = out_node
        self.in_node = in_node
        self.duplicate_str = duplicate_str
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
        self.out_edges: Dict[int, Edge] = {}
        self.count = 1

    def add_out_edge(self, other_id: int, duplicate_str: str = "") -> None:
        """Add the edge from this node to other node

        Parameters
        ----------
        other_id : int
            the terminal node id of the edge
        duplicated_str : str, optional
            the duplicated kmer represented by the edge. Defaults to ""

        Returns
        -------

        """
        if other_id in self.out_edges:
            self.out_edges[other_id].multiplicity += 1
        else:
            new_edge = Edge(self.id, other_id, duplicate_str=duplicate_str)
            self.out_edges[other_id] = new_edge


class deBruijn:
    """The de Bruijn graph class, with construction, visualization and alignment

    Attributes
    ----------
    id_count : int
        the current id of nodes added to the de Bruijn graph
    k : int
        the kmer size of the de Bruijn graph
    nodes : Dict[int, Node]
        the map from NodeID to Node object
    exist_kmer : Dict[str, int]
        currently existing kmer in de Bruijn graph, stored in form of (kmer: NodeID)
    seq_node_idx : Dict[int, List[int]]
        the map from sequence ID to the list of Node indices from that sequence
    merge_node_idx : List[int]
        the list of Node indices
    seq_last_kmer_id : List[int]
        the NodeID for last kmer in each sequence
    moltype : str
        the molecular type for characters in sequences
    names : Tuple[Any, ...]
        the sequences names
    sequences : Tuple[str, ...]
        the sequences contained in the de Bruijn graph
    num_seq : int
        number of sequences in the de Bruijn graph
    avg_len : int
        average length of sequences

    """

    def __init__(self, data: Any, k: int, moltype: str) -> None:
        """Constructor for a de Bruijn graph

        Parameters
        ----------
        data : Any
            can be path to sequences or List of sequences or SequenceCollection object
        k : int
            the kmer size for the de Bruijn graph

        Returns
        -------

        """
        sc: SequenceCollection = load_sequences(data=data, moltype=moltype)
        self.id_count = 0
        self.k = k
        self.nodes: Dict[int, Node] = {}
        self.exist_kmer: Dict[str, int] = {}
        self.seq_node_idx: Dict[int, List[int]] = {}
        self.merge_node_idx: List[int] = []
        self.seq_last_kmer_idx: List[int] = []
        self.moltype: str = moltype
        self.names: Tuple[Any, ...] = tuple(sc.names)
        self.sequences: Tuple[str, ...] = tuple([str(seq) for seq in sc.seqs])
        self.num_seq: int = len(self.sequences)
        # calculate the average sequences length
        self.avg_len = sum(map(len, self.sequences)) / self.num_seq
        self.add_debruijn()

    def _add_node(self, kmer: str = "", node_type: NodeType = NodeType.middle) -> int:
        """Add a single node to the de Bruijn graph

        Parameters
        ----------
        kmer : str, optional
            the kmer from the sequence. Defaults to ""
        node_type : NodeType, optional
            the NodeType of the new node added. Defaults to NodeType.middle

        Returns
        -------
        int
            the node id of the provided kmer

        """
        if kmer in self.exist_kmer:
            exist_node_id = self.exist_kmer[kmer]
            self.nodes[exist_node_id].count += 1
            self.merge_node_idx.append(exist_node_id)
            return exist_node_id

        new_node = Node(id=self.id_count, kmer=kmer, node_type=node_type)
        self.nodes[self.id_count] = new_node

        if node_type is NodeType.middle:
            self.exist_kmer[kmer] = self.id_count

        self.id_count += 1
        return self.id_count - 1

    def _add_edge(self, out_id: int, in_id: int, duplicate_str: str = "") -> None:
        """Connect two kmers in the de Bruijn graph with an edge

        Parameters
        ----------
        out_id : int
            the starting node id
        in_id : int
            the terminal node id
        duplicated_str : str, optional
            the duplicated kmer represented by the edge. Defaults to ""

        Returns
        -------

        """
        self.nodes[out_id].add_out_edge(in_id, duplicate_str=duplicate_str)

    def _get_seq_kmer_indices(
        self, kmers: List[str], duplicate_kmer: Set[str]
    ) -> List[int]:
        """Add kmers (of a sequence) to the de Bruijn graph, also get their indices

        Parameters
        ----------
        kmers : List[str]
            kmers of a sequence
        duplicate_kmer : Set[str]
            kmers where they are recorded to be duplicate (not included in nodes)

        Returns
        -------
        List[int]
            the indices of nodes where kmers are added

        """
        # to record the duplicated kmers
        edge_kmer = []
        # starting node
        prev_idx = self._add_node(kmer="", node_type=NodeType.start)
        node_indices = [prev_idx]
        for kmer in kmers:
            if kmer in duplicate_kmer:
                edge_kmer.append(kmer)
                current_idx = prev_idx
            elif len(edge_kmer) > 0:
                # if there are duplicated kmers, store them in the edge
                current_idx = self._add_node(kmer)
                node_indices.append(current_idx)
                self._add_edge(prev_idx, current_idx, "".join(edge_kmer))
                edge_kmer = []
            else:
                # if there is no duplicated kmer, add new nodes
                current_idx = self._add_node(kmer)
                node_indices.append(current_idx)
                self._add_edge(prev_idx, current_idx)

            # store the previous node index
            prev_idx = current_idx

        if not edge_kmer:
            self.seq_last_kmer_idx.append(current_idx)
        else:
            # if -1, the last kmer is shown as edge, should be read fully
            self.seq_last_kmer_idx.append(-1)
        # create the terminal node and connect with the previous node
        end_node = self._add_node(kmer="", node_type=NodeType.end)
        node_indices.append(end_node)
        self._add_edge(current_idx, end_node, "".join(edge_kmer))
        # return the sequence kmer indices
        return node_indices

    def add_debruijn(self) -> None:
        """Construct de Bruijn graph for the given sequences

        Parameters
        ----------

        Returns
        -------

        """
        assert self.num_seq == len(self.sequences) == 2
        # get kmers
        kmer_seqs = [get_kmers(seq, self.k) for seq in self.sequences]
        # find the duplicate kmers in each sequence
        duplicate_kmer = duplicate_kmers(kmer_seqs=kmer_seqs)
        # add nodes to de Bruijn graph
        for i, kmer_seq in enumerate(kmer_seqs):
            self.seq_node_idx[i] = self._get_seq_kmer_indices(
                kmers=kmer_seq, duplicate_kmer=duplicate_kmer
            )

    def visualize(self, path: str, save_dot: bool = False) -> None:  # pragma: no cover
        """Visualize the de Bruijn graph

        Parameters
        ----------
        path : str
            the path to the output image file
        save_dot : bool, optional
            whether save the DOT representation to file. Defaults to False.

        Returns
        -------

        Raises
        ------
        ValueError
            when the image file extension is not recognized

        """
        file_path = Path(path).expanduser().absolute()
        suffix = file_path.suffix[1:]
        if suffix not in ["pdf", "png", "svg"]:
            raise ValueError("Not supported file format")
        digraph = to_DOT(list(self.nodes.values()))
        # output the image
        digraph.render(outfile=file_path, format=suffix, cleanup=True)
        if save_dot:
            dot_file_path = file_path.with_suffix(".DOT")
            digraph.save(dot_file_path)

    def read_from_kmer(self, node_idx: int, seq_idx: int) -> str:
        """Read nucleotide(s) from the kmer with provided index

        Parameters
        ----------
        node_idx : int
            the node index of the kmer to read
        seq_idx : int
            the sequence index from kmer

        Returns
        -------
        str
            the nucleotide(s) read from the kmer

        """
        assert seq_idx in {0, 1}
        kmer = self.nodes[node_idx].kmer
        return kmer if node_idx == self.seq_last_kmer_idx[seq_idx] else kmer[0]

    def extract_bubble(self) -> List[List[Union[int, List[int]]]]:
        """Extract indicies of bubbles and merge nodes of sequences in the de Bruijn graph

        Returns
        -------
        List[List[Union[int, List[int]]]]
            List of indicies of bubbles and merge nodes of sequences in the de Bruijn graph
        """
        expansion = []
        for seq_idx in range(self.num_seq):
            bubbles = []
            bubble = []
            beginning = True
            for node_idx in self.seq_node_idx[seq_idx]:
                if self.nodes[node_idx].node_type not in {NodeType.start, NodeType.end}:
                    if node_idx not in self.merge_node_idx:
                        bubble.append(node_idx)
                    else:
                        if bubble or beginning:
                            bubbles.append(bubble)
                            bubble = []
                            beginning = False
                        bubbles.append(node_idx)
            bubbles.append(bubble)
            expansion.append(bubbles)
        return expansion

    def extract_bubble_seq(self, bubble_idx_seq: List[int], seq_idx: int) -> str:
        """Extract the string from the bubble with indices of nodes

        Parameters
        ----------
        bubble_idx_seq : List[int]
            indices of nodes that are in the bubble
        seq_idx : int
            the sequence index the bubble belongs to

        Returns
        -------
        str
            the string from the bubble with indices of nodes

        """
        assert seq_idx in {0, 1}
        rtn = []
        for i in range(len(bubble_idx_seq) - 1):
            node_idx = bubble_idx_seq[i]
            if self.nodes[node_idx].node_type not in [NodeType.start, NodeType.end]:
                rtn.append(self.read_from_kmer(node_idx, seq_idx))
            next_node_idx = bubble_idx_seq[i + 1]
            # if the next node is the end of sequence, read edge fully,
            # otherwise, read the first char
            edge_kmer = self.nodes[node_idx].out_edges[next_node_idx].duplicate_str
            if self.nodes[next_node_idx].node_type is NodeType.end:
                rtn.append(edge_kmer)
            else:
                rtn.append(read_nucleotide_from_kmers(edge_kmer, self.k))

        # if the next node is not end node, then it should be a merge node
        # last_node_idx = bubble_idx_seq[-1]
        # if not self.nodes[last_node_idx].node_type is NodeType.end:
        #     rtn.append(self._read_from_kmer(last_node_idx, seq_idx=seq_idx))
        return "".join(rtn)

    def bubble_aln(
        self,
        bubble_indices_seq1: List[int],
        bubble_indices_seq2: List[int],
        s: Dict[Tuple[str, str], int],
        d: int,
        e: int,
        prev_edge_read1: str = "",
        prev_edge_read2: str = "",
        prev_merge: str = "",
    ) -> Tuple[str, str]:
        """Align the bubbles in the de Bruijn graph

        Parameters
        ----------
        bubble_idx_seq1 : List[int]
            the list containing indices of nodes of seq1 in the bubble
        bubble_idx_seq2 : List[int]
            the list containing indices of nodes of seq2 in the bubble
        s : Dict[Tuple[str, str], int]
            the DNA scoring matrix
        d : int
            gap open costs
        e : int
            gap extend costs
        prev_edge_read1 : str, optional
            the edge kmer from the edge of the last merge node for seq1. Defaults to "".
        prev_edge_read2 : str, optional
            the edge kmer from the edge of the last merge node for seq2. Defaults to "".
        prev_merge : str, optional
            the previous merge node nucleotide. Defaults to "".

        Returns
        -------
        Tuple[str, str]
            the aligned sequences from the bubble in the de Bruijn graph

        """
        # short cut for faster experiments
        if (
            len(bubble_indices_seq1) == len(bubble_indices_seq2) == 1
            and prev_edge_read1 == prev_edge_read2 == ""
        ):
            return prev_merge, prev_merge

        # extract original bubble sequences
        bubble_seq1_str = f"{prev_merge}{prev_edge_read1}{self.extract_bubble_seq(bubble_indices_seq1, 0)}"
        bubble_seq2_str = f"{prev_merge}{prev_edge_read2}{self.extract_bubble_seq(bubble_indices_seq2, 1)}"

        # call the global_aln function to compute the global alignment of two sequences
        return dna_global_aln(bubble_seq1_str, bubble_seq2_str, s=s, d=d, e=e)

    def get_merge_edge(
        self,
        seq1_curr_idx: int,
        seq1_next_idx: int,
        seq2_curr_idx: int,
        seq2_next_idx: int,
    ) -> Tuple[str, str]:
        """To get the duplicate_kmer from the edge that is from the merge node

        Parameters
        ----------
        seq1_curr_idx : int
            current node index of sequence1
        seq1_next_idx : int
            next node index of sequence1
        seq2_curr_idx : int
            current node index of sequence2
        seq2_next_idx : int
            next node index of sequence2

        Returns
        -------
        Tuple[str, str]
            the duplicate_kmer string for each sequence that is from the merge node

        """
        seq1_edge_kmer = (
            self.nodes[seq1_curr_idx].out_edges[seq1_next_idx].duplicate_str
        )
        seq2_edge_kmer = (
            self.nodes[seq2_curr_idx].out_edges[seq2_next_idx].duplicate_str
        )
        # no chance that next node is End NodeType, we won't call this function for the last merge node
        merge_edge_read_seq1 = read_nucleotide_from_kmers(seq1_edge_kmer, self.k)
        merge_edge_read_seq2 = read_nucleotide_from_kmers(seq2_edge_kmer, self.k)
        return merge_edge_read_seq1, merge_edge_read_seq2


def to_alignment(
    dbg: deBruijn,
    match: int = 10,
    transition: int = -1,
    transversion: int = -8,
    d: int = 10,
    e: int = 2,
) -> str:
    """Use de Bruijn graph to align two sequences

    Parameters
    ----------
    match : int
        score for two matching nucleotide
    transition : int
        cost for DNA transition mutation
    transversion : int
        cost for DNA transversion mutation
    d : int
        gap open costs. Defaults to 10
    e : int
        gap extend costs. Defaults to 2

    Returns
    -------
    str
        the fasta representation of the alignment result

    """
    # scoring dict for aligning bubbles
    s = make_dna_scoring_dict(
        match=match, transition=transition, transversion=transversion
    )

    # if there's no merge node, apply global alignment
    if not dbg.merge_node_idx:
        alignment = dna_global_aln(
            str(dbg.sequences[0]), str(dbg.sequences[1]), s=s, d=d, e=e
        )
        return alignment_to_fasta(
            {dbg.names[0]: alignment[0], dbg.names[1]: alignment[1]}
        )

    seq1_idx, seq2_idx = 0, 0
    seq1_res, seq2_res = [], []
    merge_edge_read_seq1, merge_edge_read_seq2 = "", ""

    for i, merge_idx in enumerate(dbg.merge_node_idx):
        bubble_idx_seq1, bubble_idx_seq2 = [], []
        while (
            seq1_idx < len(dbg.seq_node_idx[0])
            and dbg.seq_node_idx[0][seq1_idx] != merge_idx
        ):
            bubble_idx_seq1.append(dbg.seq_node_idx[0][seq1_idx])
            seq1_idx += 1

        while (
            seq2_idx < len(dbg.seq_node_idx[1])
            and dbg.seq_node_idx[1][seq2_idx] != merge_idx
        ):
            bubble_idx_seq2.append(dbg.seq_node_idx[1][seq2_idx])
            seq2_idx += 1

        # if index overflow here, it must be the edge case, call fix function
        if seq1_idx == len(dbg.seq_node_idx[0]) or seq2_idx == len(dbg.seq_node_idx[1]):
            print("Fix starts")
            # a heuristic way to determine the window size
            dbg.merge_node_idx = merge_indices_fix(dbg)
            print("Fix ends")
            return to_alignment(dbg)

        # add the merge node index to the list
        bubble_idx_seq1.append(merge_idx)
        bubble_idx_seq2.append(merge_idx)

        prev_merge: str = (
            dbg.read_from_kmer(dbg.merge_node_idx[i - 1], 0) if i > 0 else ""
        )

        # get alignment of the bubble (with previous merge node nucleotide)
        bubble_alignment = dbg.bubble_aln(
            bubble_indices_seq1=bubble_idx_seq1,
            bubble_indices_seq2=bubble_idx_seq2,
            s=s,
            d=d,
            e=e,
            prev_edge_read1=merge_edge_read_seq1,
            prev_edge_read2=merge_edge_read_seq2,
            prev_merge=prev_merge,
        )

        aln_seq1 = bubble_alignment[0][1:] if i > 0 else bubble_alignment[0]
        aln_seq2 = bubble_alignment[1][1:] if i > 0 else bubble_alignment[1]

        if i != len(dbg.merge_node_idx) - 1:
            # extract the duplicate kmer on the merge node out edge
            merge_edge_read_seq1, merge_edge_read_seq2 = dbg.get_merge_edge(
                seq1_curr_idx=dbg.seq_node_idx[0][seq1_idx],
                seq1_next_idx=dbg.seq_node_idx[0][seq1_idx + 1],
                seq2_curr_idx=dbg.seq_node_idx[1][seq2_idx],
                seq2_next_idx=dbg.seq_node_idx[1][seq2_idx + 1],
            )

            seq1_res.extend([aln_seq1, dbg.read_from_kmer(merge_idx, 0)])
            seq2_res.extend([aln_seq2, dbg.read_from_kmer(merge_idx, 1)])
        else:
            # only put the bubble alignment to the result, leave the merge node to further alignment
            seq1_res.append(aln_seq1)
            seq2_res.append(aln_seq2)

            # store the last merge node
            bubble_idx_seq1, bubble_idx_seq2 = [merge_idx], [merge_idx]

        seq1_idx += 1
        seq2_idx += 1

    while seq1_idx < len(dbg.seq_node_idx[0]):
        bubble_idx_seq1.append(dbg.seq_node_idx[0][seq1_idx])
        seq1_idx += 1

    while seq2_idx < len(dbg.seq_node_idx[1]):
        bubble_idx_seq2.append(dbg.seq_node_idx[1][seq2_idx])
        seq2_idx += 1

    bubble_alignment = dbg.bubble_aln(
        bubble_indices_seq1=bubble_idx_seq1,
        bubble_indices_seq2=bubble_idx_seq2,
        s=s,
        d=d,
        e=e,
    )

    seq1_res.append(bubble_alignment[0])
    seq2_res.append(bubble_alignment[1])

    return alignment_to_fasta(
        {dbg.names[0]: "".join(seq1_res), dbg.names[1]: "".join(seq2_res)}
    )


@click.command()  # pragma: no cover
@click.option(
    "--infile", type=str, required=True, help="input unaligned sequences file"
)
@click.option(
    "--outfile", type=str, required=True, help="output aligned file destination"
)
@click.option("--k", type=int, required=True, help="kmer size")
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
def cli(infile, outfile, k, moltype, match, transition, transversion, d, e):
    dbg = deBruijn(infile, k, moltype)
    aln = to_alignment(dbg, match, transition, transversion, d, e)
    out_path = Path(outfile)
    with open(f"{out_path.stem}_k{k}{out_path.suffix}", "w") as f:
        f.write(aln)


if __name__ == "__main__":  # pragma: no cover
    cli()
