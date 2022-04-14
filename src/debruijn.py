from enum import Enum, auto
from typing import List, Set, Tuple
from collections import Counter
from pathlib import Path
import subprocess
import itertools

import numpy as np
from cogent3.align import global_pairwise, make_dna_scoring_dict
from cogent3 import load_unaligned_seqs, make_unaligned_seqs
from cogent3 import SequenceCollection


# scoring dict for aligning bubbles
s = make_dna_scoring_dict(10, -1, -8)


# longest common subsequence algorithm
# TODO: Improve memory usage using Hirschberg algorithm
def lcs(list1, list2):
    score = np.zeros((len(list1) + 1, len(list2) + 1), dtype=np.int_)
    for i, j in itertools.product(range(1, len(list1) + 1), range(1, len(list2) + 1)):
        score[i, j] = score[i - 1, j - 1] + 1 \
            if list1[i - 1] == list2[j - 1] else max(score[i - 1, j], score[i, j - 1])

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


# load the sequences and transform it into numpy array of strings (in Unicode)
def load_sequences(sequences, moltype: str = 'dna') -> np.ndarray:
    if isinstance(sequences, str):
        path = Path(sequences).expanduser().absolute()
        sequences = load_unaligned_seqs(path, moltype=moltype)
    if isinstance(sequences, SequenceCollection):
        seqs_lst = [str(seq) for seq in sequences.iter_seqs()]
        return np.array(seqs_lst)
    elif isinstance(sequences, list) and all(isinstance(elem, str) for elem in sequences):
        return np.array(sequences)
    else:
        raise ValueError('Invalid input type for sequence argument')


# find the kmers in sequences
def get_kmers(sequence: str, k: int) -> List[str]:
    if k < 0 or k > len(sequence):
        raise ValueError('Invalid k size for kmers')
    return [sequence[i: i + k] for i in range(len(sequence) - k + 1)]


# find the duplicated kmers in each sequence
def duplicate_kmers(kmer_seqs: List[List[str]]) -> Set[str]:
    duplicate_set = set()
    for kmer_seq in kmer_seqs:
        counter = Counter(kmer_seq)
        for key, value in counter.items():
            if value > 1:
                duplicate_set.add(key)
    return duplicate_set


# Enum type of different NodeTypes
class NodeType(Enum):
    start = auto()
    middle = auto()
    end = auto()


class Edge:
    """The Edge class to connect two nodes"""

    def __init__(self, out_node: int, in_node: int, duplicate_str: str = '') -> None:
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

    def add_out_edge(self, other_id: int, duplicate_str: str = '') -> None:
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


class deBruijn:
    """The de Bruijn graph class, with many useful method"""

    def __init__(self, sequences, k: int) -> None:
        """Constructor for a de Bruijn graph

        Args:
            sequence (Any): can be path to sequences or List of sequences or SequenceCollection objects
            k (int): the kmer size for the de Bruijn graph
        """
        self.id_count = 0
        self.k = k
        self.nodes = {}
        self.exist_kmer = {}
        self.seq_node_idx = {}
        self.merge_node_idx = []
        self.seq_last_kmer_idx = []
        self.sequences = load_sequences(sequences=sequences)
        self.num_seq = len(self.sequences)
        self.add_debruijn()

    def _add_node(self, kmer: str = '', node_type: NodeType = NodeType.middle) -> int:
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
        elif node_type not in [NodeType.start, NodeType.end]:
            raise ValueError('Invalid input type for sequence argument')

        self.id_count += 1
        return self.id_count - 1

    def _add_edge(self, out_id: int, in_id: int, duplicate_str: str = '') -> None:
        """Connect two kmers in the de Bruijn graph with an edge

        Args:
            out_id (int): the starting node id
            in_id (int): the terminal node id
            duplicated_str (str, optional): the duplicated kmer represented by the edge. Defaults to ''
        """
        self.nodes[out_id].add_out_edge(in_id, duplicate_str=duplicate_str)

    def _get_seq_kmer_idx(self, kmers: List[str], duplicate_kmer: Set[str]):
        # to record the duplicated kmers
        cycle_kmer = []
        # starting node
        new_node = self._add_node(kmer='', node_type=NodeType.start)
        node_idx = [new_node]
        for kmer in kmers:
            # store the previous node index
            prev_node = new_node
            if kmer in duplicate_kmer:
                cycle_kmer.append(kmer)
                new_node = prev_node
            elif len(cycle_kmer) > 0:
                # if there are duplicated kmers, store them in the edge
                new_node = self._add_node(kmer)
                node_idx.append(new_node)
                self._add_edge(prev_node, new_node, ''.join(cycle_kmer))
                cycle_kmer = []
            else:
                # if there is no duplicated kmer, add new nodes
                new_node = self._add_node(kmer)
                node_idx.append(new_node)
                self._add_edge(prev_node, new_node)
        if not cycle_kmer:
            self.seq_last_kmer_idx.append(new_node)
        else:
            # if -1, the last kmer is shown as edge, should be read fully
            self.seq_last_kmer_idx.append(-1)
        # create the terminal node and connect with the previous node
        end_node = self._add_node(kmer='', node_type=NodeType.end)
        node_idx.append(end_node)
        self._add_edge(new_node, end_node, ''.join(cycle_kmer))
        # return the sequence kmer indices
        return node_idx

    def add_debruijn(self) -> None:
        """Construct de Bruijn graph for the given sequences

        Args:
            seqs (SequenceCollection): The SequenceCollection object that contains the sequences for de Bruijn graph
        """
        assert self.num_seq == len(self.sequences) == 2
        # get kmers
        kmer_seqs = [
            get_kmers(seq, self.k)
            for seq in self.sequences
        ]
        # find the duplicate kmers in each sequence
        duplicate_kmer = duplicate_kmers(kmer_seqs=kmer_seqs)
        # add nodes to de Bruijn graph
        for i, kmers in enumerate(kmer_seqs):
            self.seq_node_idx[i] = self._get_seq_kmer_idx(
                kmers=kmers,
                duplicate_kmer=duplicate_kmer
            )

    def _nodes_DOT_repr(self) -> str:
        """Get the DOT representation of nodes in de Bruijn graph

        Returns:
            str: the DOT representation of nodes in original graph
        """
        rtn = [
            f'\t{node.id} [label="{node.kmer}"];\n' for node in self.nodes.values()]
        return "".join(rtn)

    def _edges_DOT_repr(self) -> str:
        """Get the DOT representation of edges in de Bruijn graph

        Returns:
            str: the DOT representation of edges in original graph
        """
        rtn = []
        for index, node in enumerate(self.nodes.values()):
            for other_node, edge in node.out_edges.items():
                current_row = f'\t{node.id} -> {other_node} \
                    [label="{edge.duplicate_str}", weight={edge.multiplicity}];'
                if index != len(self.nodes) - 1:
                    current_row += '\n'
                rtn.append(current_row)
        return ''.join(rtn)

    def _to_DOT(self, path: Path) -> None:
        """Write the DOT representation to the file

        Args:
            path (Path): the Path object points to the DOT file directory
        """
        with open(path, 'w', encoding='utf-8') as f:
            f.write("digraph debuijn {\n")
            f.write(self._nodes_DOT_repr())
            f.write(self._edges_DOT_repr())
            f.write('}')

    def visualize(self, path: str, cleanup: bool = False) -> None:
        """Visualize the de Bruijn graph

        Args:
            path (str): the path points to the image file
            cleanup (bool, optional): whether delete DOT intermediate file. Defaults to False.

        Raises:
            ValueError: when the image extension is not recognized
        """
        file_path = Path(path).expanduser().absolute()
        suffix = file_path.suffix[1:]
        if suffix not in ['pdf', 'png', 'svg']:
            raise ValueError("Not supported file format")
        dot_file = file_path.with_suffix(".DOT")
        self._to_DOT(dot_file)
        subprocess.run(
            f'dot -T{suffix} {dot_file.__str__()} -o {file_path.__str__()}',
            shell=True,
            check=True
        )
        if cleanup:
            dot_file.unlink()

    def _read_kmer(self, node_idx: int, seq_idx: int) -> str:
        kmer = self.nodes[node_idx].kmer
        return kmer if node_idx == self.seq_last_kmer_idx[seq_idx] else kmer[0]

    # extract bubble kmers from the indices of nodes
    def _extract_bubble(self, bubble_idx_seq: List[int], seq_idx: int) -> str:
        rtn = []
        for i in range(len(bubble_idx_seq) - 1):
            node_idx = bubble_idx_seq[i]
            if self.nodes[node_idx].node_type not in [NodeType.start, NodeType.end]:
                rtn.append(self._read_kmer(node_idx, seq_idx))
            next_node_idx = bubble_idx_seq[i + 1]
            # if the next node is the end of sequence, read edge fully,
            # otherwise, read the first char
            if self.nodes[next_node_idx].node_type is NodeType.end:
                rtn.append(
                    self.nodes[node_idx].out_edges[next_node_idx].duplicate_str)
            else:
                edge_kmer = self.nodes[node_idx].out_edges[next_node_idx].duplicate_str
                rtn.append(
                    edge_kmer[0] if len(edge_kmer) > 0 else edge_kmer)
        return ''.join(rtn)

    # function to align the sequences in bubbles with node indices provided
    def _bubble_aln(self, bubble1, bubble2):
        bubble_seq1 = self._extract_bubble(bubble1, 0)
        bubble_seq2 = self._extract_bubble(bubble2, 1)

        if bubble_seq1 and bubble_seq2:
            seq_colllection = make_unaligned_seqs(
                {0: bubble_seq1, 1: bubble_seq2}, moltype='dna')
            partial_aln = global_pairwise(*seq_colllection.seqs, s, 10, 2)
            return str(partial_aln.seqs[0]), str(partial_aln.seqs[1])
        elif bubble_seq1:
            return bubble_seq1, '-' * len(bubble_seq1)
        elif bubble_seq2:
            return '-' * len(bubble_seq2), bubble_seq2
        else:
            return '', ''

    def to_Alignment(self) -> Tuple[str]:
        seq1_idx, seq2_idx = 0, 0
        seq1_res, seq2_res = [], []
        for merge_idx in self.merge_node_idx:
            bubble_idx_seq1, bubble_idx_seq2 = [], []
            while seq1_idx < len(self.seq_node_idx[0]) and self.seq_node_idx[0][seq1_idx] != merge_idx:
                bubble_idx_seq1.append(self.seq_node_idx[0][seq1_idx])
                seq1_idx += 1

            while seq2_idx < len(self.seq_node_idx[1]) and self.seq_node_idx[1][seq2_idx] != merge_idx:
                bubble_idx_seq2.append(self.seq_node_idx[1][seq2_idx])
                seq2_idx += 1

            # if index overflow here, it must be the edge case, call LCS function
            if seq1_idx == len(self.seq_node_idx[0]) or seq2_idx == len(self.seq_node_idx[1]):
                self.merge_node_idx = lcs(
                    self.seq_node_idx[0], self.seq_node_idx[1])
                return self.to_Alignment()

            bubble_idx_seq1.append(merge_idx)
            bubble_idx_seq2.append(merge_idx)

            bubble_alignment = self._bubble_aln(
                bubble_idx_seq1,
                bubble_idx_seq2
            )

            seq1_res.extend(
                [bubble_alignment[0], self._read_kmer(merge_idx, 0)])
            seq2_res.extend(
                [bubble_alignment[1], self._read_kmer(merge_idx, 1)])
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

        bubble_alignment = self._bubble_aln(bubble_idx_seq1, bubble_idx_seq2)

        seq1_res.append(bubble_alignment[0])
        seq2_res.append(bubble_alignment[1])

        return ''.join(seq1_res), ''.join(seq2_res)
