import click
from collections import Counter
from cogent3 import load_unaligned_seqs
from pathlib import Path


def kmers(seq: str, k: int):
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def kmers_larger_than_threshold(seq: str, k: int, threshold: float):
    counts = Counter(kmers(seq, k))
    duplicate = sum(value - 1 for value in counts.values())
    proportion = duplicate / (len(seq) - k + 1)
    return proportion > threshold


def calculate_k(path: Path, thresh: int, moltype: str):
    seqs = load_unaligned_seqs(path, moltype=moltype)
    current_k = 1

    for i, seq in enumerate(seqs.iter_seqs()):
        print(f"Checking {i + 1}th sequence")
        while kmers_larger_than_threshold(str(seq), current_k, threshold=thresh):
            print(f"Failed k: {current_k}")
            current_k += 1

    return current_k


@click.command()
@click.option(
    "--infile", type=str, required=True, help="input unaligned sequences file"
)
@click.option(
    "--thresh",
    type=float,
    default=0.001,
    required=False,
    help="Proportion of duplicate kmers im each sequence",
)
def cli(infile, thresh):
    path = Path(infile).expanduser().absolute()
    k = calculate_k(path=path, thresh=thresh, moltype="dna")
    print(f"The smallest k value for only {thresh * 100}% duplicate kmers: {k}")


if __name__ == "__main__":
    cli()
