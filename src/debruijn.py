from collections import Counter
from pathlib import Path
import subprocess
from typing import List, Set

from cogent3 import load_unaligned_seqs, make_unaligned_seqs
from cogent3 import SequenceCollection


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
        self.start_ids = []
        self.nodes = {}
        self.exist_kmer = {}
        if isinstance(sequence, str):
            path = Path(sequence).expanduser().absolute()
            self.sequence = load_unaligned_seqs(path)
            self.seq_node_idx = self.add_sequences(self.sequence)
        elif isinstance(sequence, SequenceCollection):
            self.sequence = sequence
            self.seq_node_idx = self.add_sequences(sequence)
        elif isinstance(sequence, list) and all(isinstance(elem, str) for elem in sequence):
            self.sequence = make_unaligned_seqs(sequence)
            self.seq_node_idx = self.add_sequences(self.sequence)
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
            self.nodes[self.exist_kmer[kmer]].count += 1
            return self.exist_kmer[kmer]
        if kmer == '$':
            self.start_ids.append(self.id_count)
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

    def add_sequences(self, seqs: SequenceCollection) -> List[List[int]]:
        """Construct de Bruijn graph for the given sequences

        Args:
            seqs (SequenceCollection): The SequenceCollection object that contains the sequences for de Bruijn graph
        """
        self.num_seq = seqs.num_seqs
        # the list of list to record the node index for each sequence
        seq_node_idx = []
        # find the duplicate kmers in each sequence and add them to de Bruijn graph
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
                    seq_node_idx.append([])
                # store the previous node index
                prev_node = new_node
                if kmer in duplicate_kmer:
                    acc += kmer
                    new_node = prev_node
                elif acc != '':
                    # if there are duplicated kmers, store them in the edge
                    new_node = self.__add_node(kmer)
                    seq_node_idx[i].append(new_node)
                    self.__add_edge(prev_node, new_node, acc)
                    acc = ''
                else:
                    # if there is no duplicated kmer, add new nodes
                    new_node = self.__add_node(kmer)
                    seq_node_idx[i].append(new_node)
                    self.__add_edge(prev_node, new_node)
                if j == len(kmers) - 1:
                    # create the terminal node and connect with the previous node
                    end_node = self.__add_node('#')
                    self.__add_edge(new_node, end_node)
        # return the indicies of nodes for each sequence
        return seq_node_idx

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

    def to_POA(self) -> None:
        pass
