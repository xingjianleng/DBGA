from collections import deque
from pathlib import Path
import subprocess

from cogent3 import load_unaligned_seqs


class Modcounter:
    """The counter object that is bounded by a maximum value"""

    def __init__(self, max_bound: int = 1) -> None:
        """The constructor for the Modcounter object

        Args:
            max_bound (int, optional): the maximum bound of the counter. Defaults to 1.
        """
        self.__counter: int = 0
        self.default = max_bound
        self.max_bound: int = max_bound

    def inc(self) -> None:
        """Increment the counter variable, rounded if going over the max_bound"""
        self.__counter = (self.__counter + 1) % self.max_bound

    def update_bound(self, new_max_bound: int) -> None:
        """Update the max_bound of the Modcounter object

        Args:
            new_max_bound (int): the new max bound of the counter
        """
        self.max_bound = new_max_bound
        # if the conter variable is out of the range, reset it to 0
        if self.__counter >= self.max_bound:
            self.__counter = 0

    def reset(self) -> None:
        """Reset the counter variable and max_bound to default value"""
        self.__counter: int = 0
        self.max_bound: int = self.default

    @property
    def counter(self) -> int:
        """Getter method for the counter variable

        Returns:
            int: the counter variable value
        """
        return self.__counter


class Edge:
    """The Edge class to connect two nodes"""

    def __init__(self, out_node: int, in_node: int) -> None:
        """Constructor for the Edge class

        Args:
            out_node (int): the node_id of the starting node
            in_node (int): the node_id of the terminal node
        """
        self.out_node = out_node
        self.in_node = in_node
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
        self.in_edges = {}
        self.out_edges = {}
        self.count = 1

    def add_in_edge(self, other_id: int) -> None:
        """Add the edge from other node to this node

        Args:
            other_id (int): the starting node id of the edge
        """
        if other_id in self.in_edges:
            self.in_edges[other_id].multiplicity += 1
        else:
            new_edge = Edge(other_id, self.id)
            self.in_edges[other_id] = new_edge

    def add_out_edge(self, other_id: int) -> None:
        """Add the edge from this node to other node

        Args:
            other_id (int): the terminal node id of the edge
        """
        if other_id in self.out_edges:
            self.out_edges[other_id].multiplicity += 1
        else:
            new_edge = Edge(self.id, other_id)
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

    def __add_node(self, kmer: str, start: bool = False) -> int:
        """Add a single node to the de Bruijn graph

        Args:
            kmer (str): the kmer from the sequence
            start (bool, optional): whether it is the first kmer from the sequence. Defaults to False.

        Returns:
            int: the node id of the provided kmer
        """
        if kmer in self.exist_kmer:
            self.nodes[self.exist_kmer[kmer]].count += 1
            return self.exist_kmer[kmer]
        if start:
            self.start_ids.append(self.id_count)
        new_node = Node(self.id_count, kmer)
        self.nodes[self.id_count] = new_node
        self.exist_kmer[kmer] = self.id_count
        self.id_count += 1
        return self.id_count - 1

    def __add_edge(self, out_id: int, in_id: int) -> None:
        """Connect two kmers in the de Bruijn graph with an edge

        Args:
            out_id (int): the starting node id
            in_id (int): the terminal node id
        """
        self.nodes[out_id].add_out_edge(in_id)
        self.nodes[in_id].add_in_edge(out_id)

    def add_sequence(self, sequence: str) -> None:
        """Add a sequence to the de Bruijn graph

        Args:
            sequence (str): the sequence provided
        """
        # there should be (n - k + 1) kmers, tackle two kmers each time
        index = 0
        kmer = str(sequence[index: index + self.k])
        next_id = self.__add_node(kmer, True)
        while index < len(sequence) - self.k:
            prev_id = next_id
            kmer = str(sequence[index + 1: index + self.k + 1])
            next_id = self.__add_node(kmer)
            self.__add_edge(prev_id, next_id)
            index += 1

    def load_sequences(self, path: str, moltype: str = 'dna') -> None:
        """Load sequences from a fasta file and add sequences to the de Bruijn graph

        Args:
            path (str): path to the fasta file
            moltype (str, optional): molecular type of the sequence. Defaults to 'dna'.
        """
        fasta_path = Path(path).expanduser().absolute()
        seqs = load_unaligned_seqs(fasta_path, moltype=moltype)
        for seq in seqs.iter_seqs():
            self.add_sequence(seq)

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
        for node in self.nodes.values():
            rtn.extend(
                f'\t{node.id} -> {other_node} [label="{node.kmer[1:]}", weight={edge.multiplicity}];\n'
                for other_node, edge in node.out_edges.items()
            )
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
            subprocess.run(
                f'rm {dot_file.__str__()}',
                shell=True,
                check=True
            )

    def to_POA(self) -> None:
        """Map the de Bruij graph to a partial order graph"""
        node_queue = deque(self.start_ids)
        branch = False
        terminal_branch = False
        branch_buffer = []
        index = Modcounter()
        # if graph starts with branches, add them to the branch list
        if len(self.start_ids) > 1:
            branch = True
            index.update_bound(2)
            while len(branch_buffer) < 2:
                branch_buffer.append([])

        # using breadth first search
        while len(node_queue) != 0:
            # pop the node_id from the queue
            curr_node_id = node_queue.popleft()

            if branch and self.nodes[curr_node_id].count == 2:
                # if currently in two branches and we know we are getting out of branch
                node_queue.append(curr_node_id)
            else:
                # add the successors to the queue
                for successor in self.nodes[curr_node_id].out_edges.keys():
                    node_queue.append(successor)

            # TODO: the part to write a partial order graph
            if not self.nodes[curr_node_id].out_edges:
                # all nucleiotides are required for terminal node
                # if branch is open, we will have a way to do the comparison
                if branch:
                    branch_buffer[index.counter].append(
                        self.nodes[curr_node_id].kmer)
                    terminal_kmer = ''.join(branch_buffer[index.counter])
                    index.update_bound(1)
                    del branch_buffer[index.counter]
                    terminal_branch = True
                    branch = False
                elif terminal_branch:
                    # TODO: Consider how we align two sequences here (add to partial order graph)
                    print(terminal_kmer)
                    branch_buffer[index.counter].append(
                        self.nodes[curr_node_id].kmer)
                    print(''.join(branch_buffer[index.counter]))
                    branch_buffer = []
                    index.reset()
                else:
                    # if there is no branch in the end, two sequences are aligned
                    print(self.nodes[curr_node_id].kmer)
            elif not branch and not terminal_branch:
                # if ther isn't any branch, take the normal nucleiotide
                print(self.nodes[curr_node_id].kmer[0])
            else:
                # cumulate all branch nodes and tackle them in merge part
                branch_buffer[index.counter].append(
                    self.nodes[curr_node_id].kmer[0])
                index.inc()

            # check whether we need to merge the branches
            if branch:
                merge_node = node_queue[0]
                if self.nodes[merge_node].count == 2 and merge_node == node_queue[1]:
                    branch = False
                    node_queue.popleft()

            # TODO: extract the branch contents and add them to partial order graph
            if not branch and not terminal_branch and branch_buffer:
                print(''.join(branch_buffer[0]))
                print(''.join(branch_buffer[1]))
                # reset the branch collector
                branch_buffer = []
                index.reset()

            # if there is any branches (more than one out_edges), add the number of branches
            if len(self.nodes[curr_node_id].out_edges) > 1:
                branch = True
                index.update_bound(2)
                while len(branch_buffer) < 2:
                    branch_buffer.append([])
