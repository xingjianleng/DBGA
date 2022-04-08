from typing import List, Set, Tuple
from collections import Counter
from pathlib import Path
import subprocess

from cogent3.align import global_pairwise, make_dna_scoring_dict
from cogent3 import load_unaligned_seqs, make_unaligned_seqs
from cogent3 import SequenceCollection


# the scoring dict for aligning bubbles
s = make_dna_scoring_dict(10, -1, -8)


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

    def __init__(self, id: int, kmer: str) -> None:
        """Constructor for Node class

        Args:
            id (int): the id of the node
            kmer (str): the kmer string of the node
        """
        self.id = id
        self.kmer = kmer
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

    def __init__(self, sequence, k: int) -> None:
        """Constructor for a de Bruijn graph

        Args:
            sequence (Any): can be path to sequences or List of sequences or SequenceCollection objects
            k (int): the kmer size for the de Bruijn graph
        """
        self.id_count = 0
        self.k = k
        self.num_seq = 0
        self.nodes = {}
        self.exist_kmer = {}
        self.seq_node_idx = {}
        self.merge_node_idx = set()
        self.seq_end_idx = []
        if isinstance(sequence, str):
            path = Path(sequence).expanduser().absolute()
            self.sequence = load_unaligned_seqs(path)
            self.add_sequences(self.sequence)
        elif isinstance(sequence, SequenceCollection):
            self.sequence = sequence
            self.add_sequences(self.sequence)
        elif isinstance(sequence, list) and all(isinstance(elem, str) for elem in sequence):
            self.sequence = make_unaligned_seqs(
                {num: sequence[num] for num in range(len(sequence))})
            self.add_sequences(self.sequence)
        else:
            raise ValueError('Invalid input type for sequence argument')

    def __add_node(self, kmer: str) -> int:
        """Add a single node to the de Bruijn graph

        Args:
            kmer (str): the kmer from the sequenceDefaults to False.

        Returns:
            int: the node id of the provided kmer
        """
        if kmer in self.exist_kmer:
            exist_node_id = self.exist_kmer[kmer]
            self.nodes[exist_node_id].count += 1
            # FIXME: If edge cases happen, the merge list is not the LCS of two node_idx list
            self.merge_node_idx.add(exist_node_id)
            return exist_node_id
        new_node = Node(self.id_count, kmer)
        self.nodes[self.id_count] = new_node
        if kmer not in ['$', '#']:
            self.exist_kmer[kmer] = self.id_count
        self.id_count += 1
        return self.id_count - 1

    def __add_edge(self, out_id: int, in_id: int, duplicate_str: str = '') -> None:
        """Connect two kmers in the de Bruijn graph with an edge

        Args:
            out_id (int): the starting node id
            in_id (int): the terminal node id
            duplicated_str (str, optional): the duplicated kmer represented by the edge. Defaults to ''
        """
        self.nodes[out_id].add_out_edge(in_id, duplicate_str=duplicate_str)

    def __kmers(self, sequence: str, duplicate_kmer: Set, k: int) -> List[str]:
        kmers_str = [sequence[i: i + k] for i in range(len(sequence) - k + 1)]
        counter = Counter(kmers_str)
        for key, value in counter.items():
            if value > 1:
                duplicate_kmer.add(key)
        return kmers_str

    def add_sequences(self, seqs: SequenceCollection) -> None:
        """Construct de Bruijn graph for the given sequences

        Args:
            seqs (SequenceCollection): The SequenceCollection object that contains the sequences for de Bruijn graph
        """
        self.num_seq = seqs.num_seqs
        # find the duplicate kmers in each sequence
        duplicate_kmer = set()
        # get kmers
        kmer_seqs = [
            self.__kmers(str(seq), duplicate_kmer, self.k)
            for seq in seqs.iter_seqs()
        ]
        # add nodes to de Bruijn graph
        for i, kmers in enumerate(kmer_seqs):
            # to record the duplicated kmers
            acc = ''
            for j, kmer in enumerate(kmers):
                if j == 0:
                    # starting node
                    new_node = self.__add_node("$")
                    self.seq_node_idx[i] = [new_node]
                # store the previous node index
                prev_node = new_node
                if kmer in duplicate_kmer:
                    acc += kmer
                    new_node = prev_node
                elif acc != '':
                    # if there are duplicated kmers, store them in the edge
                    new_node = self.__add_node(kmer)
                    self.seq_node_idx[i].append(new_node)
                    self.__add_edge(prev_node, new_node, acc)
                    acc = ''
                else:
                    # if there is no duplicated kmer, add new nodes
                    new_node = self.__add_node(kmer)
                    self.seq_node_idx[i].append(new_node)
                    self.__add_edge(prev_node, new_node)
                if j == len(kmers) - 1:
                    # mark the last kmer
                    if not acc:
                        self.seq_end_idx.append(new_node)
                    else:
                        # if -1, the last kmer is shown as edge, should be read fully
                        self.seq_end_idx.append(-1)
                    # create the terminal node and connect with the previous node
                    end_node = self.__add_node('#')
                    self.seq_node_idx[i].append(end_node)
                    self.__add_edge(new_node, end_node, acc)

    def __nodes_DOT_repr(self) -> str:
        """Get the DOT representation of nodes in de Bruijn graph

        Returns:
            str: the DOT representation of nodes in original graph
        """
        rtn = [
            f'\t{node.id} [label="{node.kmer}"];\n' for node in self.nodes.values()]
        return "".join(rtn)

    def __edges_DOT_repr(self) -> str:
        """Get the DOT representation of edges in de Bruijn graph

        Returns:
            str: the DOT representation of edges in original graph
        """
        rtn = []
        for index, node in enumerate(self.nodes.values()):
            for other_node, edge in node.out_edges.items():
                current_row = f'\t{node.id} -> {other_node} [label="{edge.duplicate_str}", weight={edge.multiplicity}];'
                if index != len(self.nodes) - 1:
                    current_row += '\n'
                rtn.append(current_row)
        return ''.join(rtn)

    def __to_DOT(self, path: Path) -> None:
        """Write the DOT representation to the file

        Args:
            path (Path): the Path object points to the DOT file directory
        """
        with open(path, 'w', encoding='utf-8') as f:
            f.write("digraph debuijn {\n")
            f.write(self.__nodes_DOT_repr())
            f.write(self.__edges_DOT_repr())
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
        self.__to_DOT(dot_file)
        subprocess.run(
            f'dot -T{suffix} {dot_file.__str__()} -o {file_path.__str__()}',
            shell=True,
            check=True
        )
        if cleanup:
            dot_file.unlink()

    def __read_kmer(self, node_idx, seq_idx) -> str:
        kmer = self.nodes[node_idx].kmer
        return kmer if node_idx == self.seq_end_idx[seq_idx] else kmer[0]

    # extract bubble kmers from the indices of nodes
    def __extract_bubble(self, bubble_idx_seq: List[int], seq_idx) -> str:
        rtn = []
        for i in range(len(bubble_idx_seq) - 1):
            node_idx = bubble_idx_seq[i]
            node_kmer = self.nodes[node_idx].kmer
            if node_kmer not in ['#', '$']:
                rtn.append(self.__read_kmer(node_idx, seq_idx))
            next_node_idx = bubble_idx_seq[i + 1]
            # if the next node is '#' (end of sequence), read edge fully,
            # otherwise, read the first char
            if self.nodes[next_node_idx].kmer == '#':
                rtn.append(
                    self.nodes[node_idx].out_edges[next_node_idx].duplicate_str)
            else:
                edge_kmer = self.nodes[node_idx].out_edges[next_node_idx].duplicate_str
                rtn.append(
                    edge_kmer[0] if len(edge_kmer) > 0 else edge_kmer)
        return ''.join(rtn)

    # function to align the sequences in bubbles
    def __bubble_aln(self, bubble1, bubble2):
        bubble_seq1 = self.__extract_bubble(bubble1, 0)
        bubble_seq2 = self.__extract_bubble(bubble2, 1)

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

    def to_POA(self) -> Tuple[str]:
        seq1_idx, seq2_idx = 0, 0
        seq1_res, seq2_res = [], []
        for merge in self.merge_node_idx:
            bubble_idx_seq1, bubble_idx_seq2 = [], []
            while seq1_idx < len(self.seq_node_idx[0]) and self.seq_node_idx[0][seq1_idx] != merge:
                bubble_idx_seq1.append(self.seq_node_idx[0][seq1_idx])
                seq1_idx += 1

            while seq2_idx < len(self.seq_node_idx[1]) and self.seq_node_idx[1][seq2_idx] != merge:
                bubble_idx_seq2.append(self.seq_node_idx[1][seq2_idx])
                seq2_idx += 1

            # TODO: If index overflow here, it must be the edge case, call LCS function?

            bubble_idx_seq1.append(merge)
            bubble_idx_seq2.append(merge)

            bubble_alignment = self.__bubble_aln(
                bubble_idx_seq1, bubble_idx_seq2)

            seq1_res.extend([bubble_alignment[0], self.__read_kmer(merge, 0)])
            seq2_res.extend([bubble_alignment[1], self.__read_kmer(merge, 1)])
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

        bubble_alignment = self.__bubble_aln(bubble_idx_seq1, bubble_idx_seq2)

        seq1_res.append(bubble_alignment[0])
        seq2_res.append(bubble_alignment[1])

        return ''.join(seq1_res), ''.join(seq2_res)
