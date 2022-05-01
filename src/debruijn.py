from enum import Enum, auto
from typing import List, Set, Tuple
from collections import Counter
from pathlib import Path
import itertools

import graphviz
import numpy as np
from cogent3.align import global_pairwise, make_dna_scoring_dict
from cogent3 import load_unaligned_seqs, make_unaligned_seqs
from cogent3 import SequenceCollection


# scoring dict for aligning bubbles
s = make_dna_scoring_dict(10, -1, -8)


def balancing_aln_seqs(seq1: str, seq2: str) -> Tuple[str]:
    """Helper function to compensate the edge case where one sequence has further end than the other

    Args:
        seq1 (str): the first aligned sequence
        seq2 (str): the second aligned sqeuence

    Raises:
        ValueError: get invalid aligned sequence 1
        ValueError: get invalid aligned sequence 2

    Returns:
        Tuple[str]: two aligned sequences without extra gaps at the end
    """
    if len(seq1) == len(seq2):
        return seq1, seq2
    elif len(seq1) > len(seq2):
        difference_len = len(seq1) - len(seq2)
        if seq1[-difference_len:] != "".join(difference_len * ["-"]):
            raise ValueError(
                "Input sequences are not appropriately aligned in sequence 1!"
            )
        return seq1[:-difference_len], seq2
    else:
        difference_len = len(seq2) - len(seq1)
        if seq2[-difference_len:] != "".join(difference_len * ["-"]):
            raise ValueError(
                "Input sequences are not appropriately aligned in sequence 2!"
            )
        return seq1, seq2[:-difference_len]


def read_debruijn_edge_kmer(seq: str, k: int) -> str:
    """Read the kmer(s) contained in edges in a de Bruijn graph

    Args:
        seq (str): the duplicate string sequence from edges of de Bruijn graph
        k (int): the kmer size of de Bruijn graph

    Returns:
        str: the edge kmer read
    """
    assert len(seq) % k == 0
    rtn = [seq[i] for i in range(0, len(seq), k)]
    return "".join(rtn)


# TODO: Improve memory usage using Hirschberg algorithm
def lcs(list1: list, list2: list) -> list:
    """Find the longest common subsequence of two lists

    Args:
        list1 (list): the first input list
        list2 (list): the second input list
    Returns:
        list: the list containing longest common subsequence
    """
    score = np.zeros((len(list1) + 1, len(list2) + 1), dtype=np.int_)
    for i, j in itertools.product(range(1, len(list1) + 1), range(1, len(list2) + 1)):
        score[i, j] = (
            score[i - 1, j - 1] + 1
            if list1[i - 1] == list2[j - 1]
            else max(score[i - 1, j], score[i, j - 1])
        )

    longest_length = score[len(list1), len(list2)]
    common_seq = [-1] * longest_length

    idx1 = len(list1)
    idx2 = len(list2)

    while idx1 > 0 and idx2 > 0:
        if list1[idx1 - 1] == list2[idx2 - 1]:
            common_seq[longest_length - 1] = list1[idx1 - 1]
            idx1 -= 1
            idx2 -= 1
            longest_length -= 1
        elif score[idx1 - 1][idx2] > score[idx1][idx2 - 1]:
            idx1 -= 1
        else:
            idx2 -= 1
    return common_seq


def load_sequences(data) -> np.ndarray:
    """Load the sequences and transform it into numpy array of strings (in Unicode)

    Args:
        sequences : sequences to load, could be `path`, `SequenceCollection`, `list`
        moltype (str, optional): _description_. Defaults to 'dna'.

    Raises:
        ValueError: if the sequences parameter is not `path`, `SequenceCollection`, `list`

    Returns:
        np.ndarray: the numpy array containing the loaded sequences
    """
    moltype = "dna"
    if isinstance(data, str):
        path = Path(data).expanduser().absolute()
        data = load_unaligned_seqs(path, moltype=moltype)
    elif isinstance(data, list) and all(isinstance(elem, str) for elem in data):
        data = make_unaligned_seqs(data, moltype=moltype)
    if isinstance(data, SequenceCollection):
        seqs_lst = [str(seq) for seq in data.iter_seqs()]
        return np.array(seqs_lst)
    else:
        raise ValueError("Invalid input for sequence argument")


def get_kmers(sequence: str, k: int) -> List[str]:
    """Get the kmers in sequences

    Args:
        sequence (str): the sequence for calculating kmers
        k (int): k size for each kmer

    Raises:
        ValueError: the k value should be in [1, len(sequence)]

    Returns:
        List[str]: the list of kmers of the sequence
    """
    if k < 0 or k > len(sequence):
        raise ValueError("Invalid k size for kmers")
    return [sequence[i : i + k] for i in range(len(sequence) - k + 1)]


def duplicate_kmers(kmer_seqs: List[List[str]]) -> Set[str]:
    """Get the duplicate kmers from each sequence

    Args:
        kmer_seqs (List[List[str]]): list of kmers for each sequence

    Returns:
        Set[str]: the set containing duplicate kmers
    """
    assert len(kmer_seqs) == 2
    duplicate_set = set()
    for kmer_seq in kmer_seqs:
        counter = Counter(kmer_seq)
        for key, value in counter.items():
            if value > 1:
                duplicate_set.add(key)
    return duplicate_set


def global_aln(seq1: str, seq2: str) -> Tuple[str]:
    """Align the sequences in bubbles with node indices provided

    Args:
        seq1 (str): the first sequence to align
        seq2 (str): the second sequence to align

    Returns:
        Tuple[str]: the tuple of aligned sequences
    """
    if seq1 and seq2:
        seq_colllection = make_unaligned_seqs({0: seq1, 1: seq2}, moltype="dna")
        partial_aln = global_pairwise(*seq_colllection.seqs, s, 10, 2)
        return str(partial_aln.seqs[0]), str(partial_aln.seqs[1])
    elif seq1:
        return seq1, "-" * len(seq1)
    elif seq2:
        return "-" * len(seq2), seq2
    else:
        return "", ""


# Enum type of different NodeTypes
class NodeType(Enum):
    """NodeType class to indicate the type of node (start/middle/end)"""

    start = auto()
    middle = auto()
    end = auto()


class Edge:
    """The Edge class to connect two nodes"""

    def __init__(self, out_node: int, in_node: int, duplicate_str: str = "") -> None:
        """Constructor for the Edge class

        Args:
            out_node (int): the node_id of the starting node
            in_node (int): the node_id of the terminal node
            duplicate_str (str): the edge can represent duplicate kmers
        """
        self.out_node = out_node
        self.in_node = in_node
        self.duplicate_str = duplicate_str
        self.multiplicity = 1


class Node:
    """The Node class to construct a de Bruijn graph"""

    def __init__(self, id: int, kmer: str, node_type: NodeType) -> None:
        """Constructor for Node class

        Args:
            id (int): the id of the node
            kmer (str): the kmer string of the node
        """
        self.id = id
        self.kmer = kmer
        self.node_type = node_type
        # the edge between two nodes won't be duplicate in the same direction
        self.out_edges = {}
        self.count = 1

    def add_out_edge(self, other_id: int, duplicate_str: str = "") -> None:
        """Add the edge from this node to other node

        Args:
            other_id (int): the terminal node id of the edge
            duplicated_str (str, optional): the duplicated kmer represented by the edge. Defaults to ''
        """
        if other_id in self.out_edges:
            self.out_edges[other_id].multiplicity += 1
        else:
            new_edge = Edge(self.id, other_id, duplicate_str=duplicate_str)
            self.out_edges[other_id] = new_edge


def to_DOT(nodes: List[Node]) -> graphviz.Digraph:
    """Obtain the DOT representation to the de Bruijn graph

    Args:
        nodes (List[Node]): list of nodes that belong to a de Bruijn graph

    Returns:
        graphviz.Digraph: graphviz Digraph object representing the de Bruijn graph
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


class deBruijn:
    """The de Bruijn graph class, with many useful method"""

    def __init__(self, data, k: int) -> None:
        """Constructor for a de Bruijn graph

        Args:
            sequence (Any): can be path to sequences or List of sequences or SequenceCollection object
            k (int): the kmer size for the de Bruijn graph
        """
        self.id_count = 0
        self.k = k
        self.nodes = {}
        self.exist_kmer = {}
        self.seq_node_idx = {}
        self.merge_node_idx = []
        self.seq_last_kmer_idx = []
        self.sequences = load_sequences(data=data)
        self.num_seq = len(self.sequences)
        self.add_debruijn()

    def _add_node(self, kmer: str = "", node_type: NodeType = NodeType.middle) -> int:
        """Add a single node to the de Bruijn graph

        Args:
            kmer (str): the kmer from the sequenceDefaults to False.
            node_type (NodeType, optional): the NodeType of the new node added

        Returns:
            int: the node id of the provided kmer
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

        Args:
            out_id (int): the starting node id
            in_id (int): the terminal node id
            duplicated_str (str, optional): the duplicated kmer represented by the edge. Defaults to ''
        """
        self.nodes[out_id].add_out_edge(in_id, duplicate_str=duplicate_str)

    def _get_seq_kmer_idx(
        self, kmers: List[str], duplicate_kmer: Set[str]
    ) -> List[int]:
        """Add kmers (of a sequence) to the de Bruijn graph, also get their indices

        Args:
            kmers (List[str]): kmers of a sequence
            duplicate_kmer (Set[str]): kmers where they are recorded to be duplicate (not included in nodes)

        Returns:
            List[int]: the indices of nodes where kmers are added
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

        Args:
            seqs (SequenceCollection): The SequenceCollection object that contains the sequences for de Bruijn graph
        """
        assert self.num_seq == len(self.sequences) == 2
        # get kmers
        kmer_seqs = [get_kmers(seq, self.k) for seq in self.sequences]
        # find the duplicate kmers in each sequence
        duplicate_kmer = duplicate_kmers(kmer_seqs=kmer_seqs)
        # add nodes to de Bruijn graph
        for i, kmer_seq in enumerate(kmer_seqs):
            self.seq_node_idx[i] = self._get_seq_kmer_idx(
                kmers=kmer_seq, duplicate_kmer=duplicate_kmer
            )

    def visualize(self, path: str, save_DOT: bool = False) -> None:
        """Visualize the de Bruijn graph

        Args:
            path (str): the path to the output image file
            save_DOT (bool, optional): whether save the DOT representation to file. Defaults to False.

        Raises:
            ValueError: when the image extension is not recognized
        """
        file_path = Path(path).expanduser().absolute()
        suffix = file_path.suffix[1:]
        if suffix not in ["pdf", "png", "svg"]:
            raise ValueError("Not supported file format")
        digraph = to_DOT(list(self.nodes.values()))
        # output the image
        digraph.render(outfile=file_path, format=suffix, cleanup=True)
        if save_DOT:
            dot_file_path = file_path.with_suffix(".DOT")
            digraph.save(dot_file_path)

    def _read_from_kmer(self, node_idx: int, seq_idx: int) -> str:
        """Read nucleotide(s) from the kmer with provided index

        Args:
            node_idx (int): the node index of the kmer to read
            seq_idx (int): the sequence index from kmer

        Returns:
            str: the nucleotide(s) read from the kmer
        """
        assert seq_idx in {0, 1}
        kmer = self.nodes[node_idx].kmer
        return kmer if node_idx == self.seq_last_kmer_idx[seq_idx] else kmer[0]

    def _extract_bubble(self, bubble_idx_seq: List[int], seq_idx: int) -> str:
        """Extract the string from the bubble from indices of nodes

        Args:
            bubble_idx_seq (List[int]): indices of nodes that are in the bubble
            seq_idx (int): the sequence index the bubble belongs to

        Returns:
            str: the string from the bubble from indices of nodes
        """
        assert seq_idx in {0, 1}
        rtn = []
        for i in range(len(bubble_idx_seq) - 1):
            node_idx = bubble_idx_seq[i]
            if self.nodes[node_idx].node_type not in [NodeType.start, NodeType.end]:
                rtn.append(self._read_from_kmer(node_idx, seq_idx))
            next_node_idx = bubble_idx_seq[i + 1]
            # if the next node is the end of sequence, read edge fully,
            # otherwise, read the first char
            edge_kmer = self.nodes[node_idx].out_edges[next_node_idx].duplicate_str
            if self.nodes[next_node_idx].node_type is NodeType.end:
                rtn.append(edge_kmer)
            else:
                rtn.append(read_debruijn_edge_kmer(edge_kmer, self.k))
        return "".join(rtn)

    def _bubble_aln(
        self,
        bubble_idx_seq1: List[int],
        bubble_idx_seq2: List[int],
        edge_read1: str = "",
        edge_read2: str = "",
    ) -> Tuple[str]:
        """Align the bubbles in the de Bruijn graph

        Args:
            bubble_idx_seq1 (List[int]): the list containing indices of nodes of seq1 in the bubble
            bubble_idx_seq2 (List[int]): the list containing indices of nodes of seq2 in the bubble
            edge_read1 (str, optional): the edge kmer from the edge of the last merge node for seq1. Defaults to ''.
            edge_read2 (str, optional): the edge kmer from the edge of the last merge node for seq2. Defaults to ''.

        Returns:
            Tuple[str]: the aligned sequences from the bubble in the de Bruijn graph
        """
        # extract original bubble sequences
        bubble_seq1_str = f"{edge_read1}{self._extract_bubble(bubble_idx_seq1, 0)}"
        bubble_seq2_str = f"{edge_read2}{self._extract_bubble(bubble_idx_seq2, 1)}"
        # call the global_aln function to compute the global alignment of two sequences
        return global_aln(bubble_seq1_str, bubble_seq2_str)

    def _get_merge_edge(
        self,
        seq1_curr_idx: List[int],
        seq1_next_idx: List[int],
        seq2_curr_idx: List[int],
        seq2_next_idx: List[int],
    ) -> Tuple[str]:
        """To get the duplicate_kmer from the edge that is from the merge node

        Args:
            seq1_curr_idx (List[int]): current node index of sequence1
            seq1_next_idx (List[int]): next node index of sequence1
            seq2_curr_idx (List[int]): current node index of sequence2
            seq2_next_idx (List[int]): next node index of sequence2

        Returns:
            Tuple[str]: the duplicate_kmer string for each sequence that is from the merge node
        """
        seq1_edge_kmer = (
            self.nodes[seq1_curr_idx].out_edges[seq1_next_idx].duplicate_str
        )
        seq2_edge_kmer = (
            self.nodes[seq2_curr_idx].out_edges[seq2_next_idx].duplicate_str
        )
        if self.nodes[seq1_next_idx].node_type is NodeType.end:
            merge_edge_read_seq1 = seq1_edge_kmer
        else:
            merge_edge_read_seq1 = read_debruijn_edge_kmer(seq1_edge_kmer, self.k)
        if self.nodes[seq2_next_idx].node_type is NodeType.end:
            merge_edge_read_seq2 = seq2_edge_kmer
        else:
            merge_edge_read_seq2 = read_debruijn_edge_kmer(seq2_edge_kmer, self.k)
        return merge_edge_read_seq1, merge_edge_read_seq2

    def to_alignment(self) -> Tuple[str]:
        """Use de Bruijn graph to align two sequences

        Returns:
            Tuple[str]: the tuple of aligned sequences
        """
        seq1_idx, seq2_idx = 0, 0
        seq1_res, seq2_res = [], []
        merge_edge_read_seq1, merge_edge_read_seq2 = "", ""

        for merge_idx in self.merge_node_idx:
            bubble_idx_seq1, bubble_idx_seq2 = [], []
            while (
                seq1_idx < len(self.seq_node_idx[0])
                and self.seq_node_idx[0][seq1_idx] != merge_idx
            ):
                bubble_idx_seq1.append(self.seq_node_idx[0][seq1_idx])
                seq1_idx += 1

            while (
                seq2_idx < len(self.seq_node_idx[1])
                and self.seq_node_idx[1][seq2_idx] != merge_idx
            ):
                bubble_idx_seq2.append(self.seq_node_idx[1][seq2_idx])
                seq2_idx += 1

            # if index overflow here, it must be the edge case, call LCS function
            if seq1_idx == len(self.seq_node_idx[0]) or seq2_idx == len(
                self.seq_node_idx[1]
            ):
                self.merge_node_idx = lcs(self.seq_node_idx[0], self.seq_node_idx[1])
                return self.to_alignment()

            # add the merge node index to the list
            bubble_idx_seq1.append(merge_idx)
            bubble_idx_seq2.append(merge_idx)

            bubble_alignment = self._bubble_aln(
                bubble_idx_seq1=bubble_idx_seq1,
                bubble_idx_seq2=bubble_idx_seq2,
                edge_read1=merge_edge_read_seq1,
                edge_read2=merge_edge_read_seq2,
            )

            seq1_res.extend([bubble_alignment[0], self._read_from_kmer(merge_idx, 0)])
            seq2_res.extend([bubble_alignment[1], self._read_from_kmer(merge_idx, 1)])

            # extract the duplicate kmer on the merge node out edge
            merge_edge_read_seq1, merge_edge_read_seq2 = self._get_merge_edge(
                seq1_curr_idx=self.seq_node_idx[0][seq1_idx],
                seq1_next_idx=self.seq_node_idx[0][seq1_idx + 1],
                seq2_curr_idx=self.seq_node_idx[1][seq2_idx],
                seq2_next_idx=self.seq_node_idx[1][seq2_idx + 1],
            )
            seq1_idx += 1
            seq2_idx += 1

        # clean bubble_idx_seq
        bubble_idx_seq1, bubble_idx_seq2 = [], []

        while seq1_idx < len(self.seq_node_idx[0]):
            bubble_idx_seq1.append(self.seq_node_idx[0][seq1_idx])
            seq1_idx += 1

        while seq2_idx < len(self.seq_node_idx[1]):
            bubble_idx_seq2.append(self.seq_node_idx[1][seq2_idx])
            seq2_idx += 1

        bubble_alignment = self._bubble_aln(
            bubble_idx_seq1=bubble_idx_seq1,
            bubble_idx_seq2=bubble_idx_seq2,
            edge_read1=merge_edge_read_seq1,
            edge_read2=merge_edge_read_seq2,
        )

        seq1_res.append(bubble_alignment[0])
        seq2_res.append(bubble_alignment[1])

        return balancing_aln_seqs("".join(seq1_res), "".join(seq2_res))
