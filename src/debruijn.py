from collections import deque
from pathlib import Path
import subprocess

from cogent3 import load_unaligned_seqs


class Edge:
    def __init__(self, out_node: int, in_node: int) -> None:
        self.out_node = out_node
        self.in_node = in_node
        self.multiplicity = 1


class Node:
    def __init__(self, id: int, kmer: str) -> None:
        self.id = id
        self.kmer = kmer
        # the edge between two nodes won't be duplicate in the same direction
        self.in_edges = {}
        self.out_edges = {}
        self.count = 1

    def add_in_edge(self, other_id: int) -> None:
        if other_id in self.in_edges:
            self.in_edges[other_id].multiplicity += 1
        else:
            new_edge = Edge(other_id, self.id)
            self.in_edges[other_id] = new_edge

    def add_out_edge(self, other_id: int) -> None:
        if other_id in self.out_edges:
            self.out_edges[other_id].multiplicity += 1
        else:
            new_edge = Edge(self.id, other_id)
            self.out_edges[other_id] = new_edge


class deBruijn:
    def __init__(self, k: int) -> None:
        self.id_count = 0
        self.k = k
        self.nodes = {}
        self.exist_kmer = {}

    def __add_node(self, kmer: str) -> int:
        if kmer in self.exist_kmer:
            self.nodes[self.exist_kmer[kmer]].count += 1
            return self.exist_kmer[kmer]
        new_node = Node(self.id_count, kmer)
        self.nodes[self.id_count] = new_node
        self.exist_kmer[kmer] = self.id_count
        self.id_count += 1
        return self.id_count - 1

    def __add_edge(self, out_id: int, in_id: int) -> None:
        self.nodes[out_id].add_out_edge(in_id)
        self.nodes[in_id].add_in_edge(out_id)

    def add_sequence(self, sequence: str) -> None:
        # there should be (n - k + 1) kmers, tackle two kmers each time
        index = 0
        kmer = sequence[index: index + self.k]
        next_id = self.__add_node(kmer)
        while index < len(sequence) - self.k:
            prev_id = next_id
            kmer = sequence[index + 1: index + self.k + 1]
            next_id = self.__add_node(kmer)
            self.__add_edge(prev_id, next_id)
            index += 1

    def load_sequences(self, path: str, moltype: str = 'dna') -> None:
        fasta_path = Path(path).expanduser().absolute()
        seqs = load_unaligned_seqs(fasta_path, moltype=moltype)
        for seq in seqs.iter_seqs():
            self.add_sequence(seq)

    def __nodes_DOT_repr(self) -> str:
        rtn = [
            f'\t{node.id} [label="{node.kmer}"];\n' for node in self.nodes.values()]
        return "".join(rtn)

    def __edges_DOT_repr(self) -> str:
        rtn = []
        for node in self.nodes.values():
            rtn.extend(
                f'\t{node.id} -> {other_node} [label="{node.kmer[1:]}", weight={edge.multiplicity}];\n'
                for other_node, edge in node.out_edges.items()
            )
        return "".join(rtn)

    def __to_DOT(self, path: Path) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write("digraph debuijn {\n")
            f.write(self.__nodes_DOT_repr())
            f.write(self.__edges_DOT_repr())
            f.write('}')

    def visualize(self, path: str, cleanup: bool = False) -> None:
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

    def to_POA(self):
        node_queue = deque()
        branch = []
        node_queue.append(0)

        # using breadth first search
        while len(node_queue) != 0:
            # pop the node_id from the queue
            curr_node_id = node_queue.popleft()

            if branch and self.nodes[curr_node_id].count == branch[-1]:
                # if currently in two branches and we know we are getting out of branch
                node_queue.append(curr_node_id)
            else:
                # add the successors to the queue
                for successor in self.nodes[curr_node_id].out_edges.keys():
                    node_queue.append(successor)

            # TODO: the part to write a partial order graph
            # `len(node_queue) == 0`` isn't good as branch can appear as ends
            if not self.nodes[curr_node_id].out_edges:
                # all nucleiotides are required for terminal node
                print(self.nodes[curr_node_id].kmer)
            elif not branch:
                # if ther isn't any branch, take the normal nucleiotide
                print(self.nodes[curr_node_id].kmer[0])
            else:
                # shouldn't be necessary here
                # FIXME: Should cumulate all branch nodes and tackle them in merge part
                print("Attention!")

            # check whether we need to merge the branches
            if branch and len(node_queue) == branch[-1]:
                merge_node = node_queue[0]
                if self.nodes[merge_node].count == branch[-1] and all(
                    merge_node == node_queue[i] for i in range(branch[-1])
                ):
                    del branch[-1]
                    node_queue.popleft()

            # if there is any branches (more than one out_edges), add the number of branches
            if len(self.nodes[curr_node_id].out_edges) > 1:
                branch.append(self.nodes[curr_node_id].count)
