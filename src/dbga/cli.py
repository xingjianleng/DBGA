from pathlib import Path
from time import time

import click
from dbga.utils import load_sequences, distance_matrix_prediction
from dbga.debruijn_msa import deBruijnMultiSeqs
from dbga.debruijn_pairwise import deBruijnPairwise


@click.command()
@click.option(
    "--infile",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input original sequences file",
)
@click.option(
    "--outfile",
    "-o",
    type=click.Path(exists=False),
    required=True,
    help="Output aligned sequences file",
)
@click.option(
    "-k", type=int, required=True, help="The kmer size for constructing de Bruijn graph"
)
@click.option(
    "--moltype", "-m", type=str, required=True, help="The input sequence molecular type"
)
@click.option(
    "--match",
    default=10,
    type=int,
    required=False,
    help="score for two matching nucleotide (pairwise only)",
)
@click.option(
    "--transition",
    default=-1,
    type=int,
    required=False,
    help="cost for DNA transition mutation (pairwise only)",
)
@click.option(
    "--transversion",
    default=-8,
    type=int,
    required=False,
    help="cost for DNA transversion mutation (pairwise only)",
)
@click.option(
    "-d",
    default=10,
    type=int,
    required=False,
    help="costs for opening a gap (pairwise only)",
)
@click.option(
    "-e",
    default=2,
    type=int,
    required=False,
    help="costs for extending a gap (pairwise only)",
)
@click.option(
    "--model",
    type=str,
    required=False,
    default="F81",
    help="The model for multiple sequence alignment (MSA only)",
)
@click.option(
    "--indel_rate",
    type=float,
    required=False,
    default=0.01,
    help="One parameter for the progressive pair-HMM (MSA only)",
)
@click.option(
    "--indel_length",
    type=float,
    required=False,
    default=0.01,
    help="One parameter for the progressive pair-HMM (MSA only)",
)
def main(
    infile,
    outfile,
    k,
    moltype,
    match,
    transition,
    transversion,
    d,
    e,
    model,
    indel_rate,
    indel_length,
):
    # check input file format
    infile_path = Path(infile)
    outfile_path = Path(outfile)
    if (
        infile_path.suffix.lower() != ".fasta"
        or outfile_path.suffix.lower() != ".fasta"
    ):
        click.secho(
            "Expect input and output files to be `fasta` format!", err=True, fg="red"
        )
        return

    # check number of sequences in the input fasta file
    sequence_collection = load_sequences(data=infile_path, moltype=moltype)
    sequence_num = sequence_collection.num_seqs
    start_time = time()

    if sequence_num < 2:
        # too few sequences
        click.secho(
            "Too few sequences for alignment, should be at least 2!", err=True, fg="red"
        )
        return

    elif sequence_num == 2:
        # pairwise sequence alignment case
        click.secho(
            f"Number of sequences: {sequence_num}, running de Bruijn pairwise alignment",
            fg="green",
        )
        dbg = deBruijnPairwise(data=sequence_collection, k=k, moltype=moltype)
        aln = dbg.alignment(
            match=match, transition=transition, transversion=transversion, d=d, e=e
        )
        with open(outfile, "w") as f:
            f.write(aln.to_fasta())

    else:
        # multiple sequence alignment case
        click.secho(
            f"Number of sequences: {sequence_num}, running de Bruijn multiple sequence alignment",
            fg="green",
        )
        dbg = deBruijnMultiSeqs(data=sequence_collection, k=k, moltype=moltype)
        estimated_dm = distance_matrix_prediction(dbg.sc)

        aln = dbg.alignment(
            model=model,
            dm=estimated_dm,
            indel_rate=indel_rate,
            indel_length=indel_length,
        )
        with open(outfile, "w") as f:
            f.write(aln.to_fasta())

    click.secho(
        f"Alignment task finishes within {round(time() - start_time, 2)} seconds!",
        fg="green",
    )
    click.secho(f"Alignment results saved in {outfile_path.absolute()}!", fg="green")


if __name__ == "__main__":
    main()
