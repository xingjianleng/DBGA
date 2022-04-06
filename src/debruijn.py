from collections import Counter
from pathlib import Path
import subprocess

from cogent3 import load_unaligned_seqs
from cogent3 import SequenceCollection


class Edge:
    """The Edge class to connect two nodes"""

    def __init__(self, out_node: int, in_node: int, duplicate_str: str = '') -> None:
        """Constructor for the Edge class

        Args:
            out_node (int): the node_id of the starting node
            in_node (int): the node_id of the terminal node
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
        """
        if other_id in self.out_edges:
            self.out_edges[other_id].multiplicity += 1
        else:
            new_edge = Edge(self.id, other_id, duplicate_str=duplicate_str)
            self.out_edges[other_id] = new_edge


class deBruijn:
    """The de Bruijn graph class, with many useful method"""

    def __init__(self, k: int) -> None:
        """Constructor for a de Bruijn graph

        Args:
            k (int): the kmer size for the de Bruijn graph
        """
        self.id_count = 0
        self.k = k
        self.start_ids = []
        self.nodes = {}
        self.exist_kmer = {}

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
            duplicated_str (str): the duplicated kmers in the sequence
        """
        self.nodes[out_id].add_out_edge(in_id, duplicate_str=duplicate_str)

    def __kmers(self, sequence: str, k: int) -> Counter:
        kmers_str = [sequence[i: i + k] for i in range(len(sequence) - k + 1)]
        return Counter(kmers_str)

    def add_sequences(self, seqs: SequenceCollection) -> None:
        """Construct de Bruijn graph for the given sequences

        Args:
            seqs (SequenceCollection): The SequenceCollection object that contains the sequences for de Bruijn graph
        """
        # get kmer counters
        kmer_seqs = [self.__kmers(seq, self.k) for seq in seqs.iter_seqs()]
        # find the duplicate kmers in each sequence and add them to de Bruijn graph
        duplicate_kmer = set()
        for counter in kmer_seqs:
            for key, value in counter.items():
                if value > 1:
                    duplicate_kmer.add(key)
        # add nodes to de Bruijn graph
        for counter in kmer_seqs:
            acc = ''
            for i, (key, _) in enumerate(counter.items()):
                if i == 0:
                    # starting node
                    new_node = self.__add_node("$")
                prev_node = new_node
                if key in duplicate_kmer:
                    acc += key
                    new_node = prev_node
                elif acc != '':
                    new_node = self.__add_node(key)
                    self.__add_edge(prev_node, new_node, acc)
                    acc = ''
                else:
                    new_node = self.__add_node(key)
                    self.__add_edge(prev_node, new_node)
                if i == len(counter) - 1:
                    end_node = self.__add_node('#')
                    self.__add_edge(new_node, end_node)

    def load_fasta(self, path: str, moltype: str = 'dna') -> None:
        """Load sequences from a fasta file and add sequences to the de Bruijn graph

        Args:
            path (str): path to the fasta file
            moltype (str, optional): molecular type of the sequence. Defaults to 'dna'.
        """
        fasta_path = Path(path).expanduser().absolute()
        seqs = load_unaligned_seqs(fasta_path, moltype=moltype)
        self.add_sequences(seqs=seqs)

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
