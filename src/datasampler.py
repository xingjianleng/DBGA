from pathlib import Path
from random import sample
from typing import List


from cogent3.parse.genbank import RichGenbankParser
from cogent3 import load_unaligned_seqs, load_aligned_seqs, make_unaligned_seqs


def sample_random_seqs(
    path: Path, moltype: str, aligned: bool = False, k: int = 2
) -> str:
    """Randomly sample two sequences from the fasta file

    Parameters
    ----------
    path : Path
        the Path object of the fasta file
    moltype : str
        the molecular type in the sequence
    aligned : bool
        whether the input sequences are aligned or not. Defaults to False
    k : int
        number of sequences sampled. Defaults to 2

    Returns
    -------
    str
        the fasta representation of the chosen sequences

    """
    if not aligned:
        seqs = load_unaligned_seqs(path, moltype=moltype)
    else:
        seqs = load_aligned_seqs(path, moltype=moltype)
    chosen_seq_names = sample(seqs.names, k=k)
    return seqs.take_seqs(chosen_seq_names).degap().to_fasta()


def sample_given_seqs(
    path: Path, moltype: str, names: List[str], aligned: bool = False
) -> str:
    """Sample sequences from the fasta file with given names

    Parameters
    ----------
    path : Path
        the Path object of the fasta file
    moltype : str
        the molecular type in the sequence
    names: List[str]
        the names of sequences chosen
    aligned : bool
        whether the input sequences are aligned or not. Defaults to False

    Returns
    -------
    str
        the fasta representation of the chosen sequences

    """
    if not aligned:
        seqs = load_unaligned_seqs(path, moltype=moltype)
    else:
        seqs = load_aligned_seqs(path, moltype=moltype)
    return seqs.take_seqs(names).degap().to_fasta()


def gbparser(in_path: str, out_path: str) -> None:
    """Parser for genbank files

    Parameters
    ----------
    in_path: str :
        path to the input .gb file
    out_path: str :
        path to the output fasta file

    Returns
    -------

    """
    with open(in_path, "r") as infile:
        n, seq = list(RichGenbankParser(infile, moltype="dna"))[0]
    seq_collection = make_unaligned_seqs({n: seq}, moltype="dna")
    with open(out_path, "w") as f:
        f.write(seq_collection.to_fasta())


if __name__ == "__main__":
    files_count = 2
    path = Path("../data/raw/influenza.fasta")
    processed_path = Path("../data/processed")
    for i in range(files_count):
        with open(f"{processed_path}/influenza-similar{i + 1}.fasta", "w") as f:
            f.write(sample_random_seqs(path, moltype="dna"))
    # gbparser("../data/raw/mers.gb", "../data/processed/mers.fasta")
