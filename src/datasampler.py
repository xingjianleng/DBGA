from pathlib import Path
from random import choice, sample


from cogent3 import load_unaligned_seqs, make_unaligned_seqs


# TODO: Need to think of a better way to decide two sequences are similar
def is_similar_seqs_corona(name1: str, name2: str) -> bool:
    return name1[:2] == name2[:2]


def sample_similar_seqs(path: Path, moltype: str = "dna") -> dict:
    """Randomly sample two similar sequences from the fasta file

    Args:
        path (Path): the Path object of the fasta file
        moltype (str, optional): molecule type. Defaults to 'dna'.

    Returns:
        dict: dictionary of the randomly sampled sequences with {sequence_name: sequence}
    """
    seqs = load_unaligned_seqs(path, moltype=moltype)

    while True:
        first_seq_name = choice(seqs.names)
        second_seq_name_options = [
            name
            for name in seqs.names
            if name != first_seq_name and is_similar_seqs_corona(first_seq_name, name)
        ]
        if second_seq_name_options:
            break
    second_seq_name = choice(second_seq_name_options)
    return {
        first_seq_name: seqs.get_seq(first_seq_name),
        second_seq_name: seqs.get_seq(second_seq_name),
    }


def sample_random_seqs(path: Path, moltype: str = "dna") -> dict:
    """Randomly sample two sequences from the fasta file

    Args:
        path (Path): the Path object of the fasta file
        moltype (str, optional): molecule type. Defaults to 'dna'.

    Returns:
        dict: dictionary of the randomly sampled sequences with {sequence_name: sequence}
    """
    seqs = load_unaligned_seqs(path, moltype=moltype)
    chosen_seq_names = sample(seqs.names, k=2)
    return {name: str(seqs.get_seq(name)) for name in chosen_seq_names}


def dict_to_fasta(seqs: dict, moltype: str = "dna") -> str:
    """Convert a dictionary of sequences into fasta format

    Args:
        seqs (dict): dictionary containing sequences with {name: sequence}
        moltype (str, optional): molecule type. Defaults to 'dna'.

    Returns:
        str: the fasta representation of sequences
    """
    seqs_collection = make_unaligned_seqs(seqs, moltype=moltype)
    return seqs_collection.to_fasta()


if __name__ == "__main__":
    files_count = 5
    path = Path("../data/raw/corona-unaligned.fasta")
    processed_path = Path("../data/processed")
    for i in range(files_count):
        with open(f"{processed_path}/similar-{i + 1}.fasta", "w") as f:
            f.write(dict_to_fasta(sample_similar_seqs(path)))
