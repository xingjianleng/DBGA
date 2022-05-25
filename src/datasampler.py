from pathlib import Path
from random import sample


from cogent3.parse.genbank import RichGenbankParser
from cogent3 import load_unaligned_seqs, make_unaligned_seqs


def sample_random_seqs(path: Path, moltype: str) -> dict:
    """Randomly sample two sequences from the fasta file

    Parameters
    ----------
    path : Path
        the Path object of the fasta file
    moltype : str
        the molecular type in the sequence

    Returns
    -------
    dict
        dictionary of the randomly sampled sequences with {sequence_name: sequence}

    """
    seqs = load_unaligned_seqs(path, moltype=moltype)
    chosen_seq_names = sample(seqs.names, k=2)
    return {name: str(seqs.get_seq(name)) for name in chosen_seq_names}


def dict_to_fasta(seqs: dict) -> str:
    """Convert a dictionary of sequences into fasta format

    Parameters
    ----------
    seqs : dict
        dictionary containing sequences with {name: sequence}
    moltype : str
        the molecular type in the sequence

    Returns
    -------
    str
        the fasta representation of sequences

    """
    seqs_collection = make_unaligned_seqs(seqs)
    return seqs_collection.to_fasta()


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
    path = Path("../data/raw/ebola.fasta")
    processed_path = Path("../data/processed")
    for i in range(files_count):
        with open(f"{processed_path}/similar-{i + 1}.fasta", "w") as f:
            f.write(dict_to_fasta(sample_random_seqs(path, moltype="dna")))
    # gbparser("../data/raw/mers.gb", "../data/processed/mers.fasta")
