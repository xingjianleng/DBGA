from cogent3.align import make_dna_scoring_dict
from cogent3.core.alphabet import AlphabetError
from cogent3 import SequenceCollection
import pytest

from src.utils import *


def sequence_loader_checker(seqs, exp_seq1: str, exp_seq2: str):
    assert isinstance(seqs, SequenceCollection)
    assert seqs.seqs[0] == exp_seq1
    assert seqs.seqs[1] == exp_seq2


def test_read_debruijn_edge_kmer():
    # empty string test case
    seq1 = ""
    assert read_nucleotide_from_kmers(seq1, 3) == ""

    # normal case, extract first char of kmer string test
    seq2 = "AOPKFNLSDOEPKPOWSEGH"
    assert read_nucleotide_from_kmers(seq2, 4) == "AFDKS"
    seq3 = "1092831790"
    assert read_nucleotide_from_kmers(seq3, 2) == "19819"

    # length of sequence is not a multiple of k, raise error
    seq4 = "ACMDANFGL"
    with pytest.raises(AssertionError):
        read_nucleotide_from_kmers(seq4, 4)
    seq5 = "109128903"
    with pytest.raises(AssertionError):
        read_nucleotide_from_kmers(seq5, 2)


def test_load_sequences():
    # given a path, load sequences
    path = "./tests/data/gap_at_end.fasta"
    sequence_loader_checker(
        load_sequences(path, moltype="dna"), "GTACAAGCGATG", "GTACAAGCGA"
    )

    # given a SequenceCollection, load sequences
    seq1 = "ACGTCAG"
    seq2 = "GACTGTGA"
    sc = SequenceCollection({"seq1": seq1, "seq2": seq2})
    sequence_loader_checker(load_sequences(sc, moltype="dna"), seq1, seq2)

    # given a list of strings, load sequences
    seq_lst = ["ACGTCAG", "GACTGTGA"]
    sequence_loader_checker(
        load_sequences(seq_lst, moltype="dna"), seq_lst[0], seq_lst[1]
    )

    # if the provided parameter is in a wrong type, raise error
    with pytest.raises(ValueError) as e:
        load_sequences({1: "ACGTGTA"}, moltype="dna")
    exec_msg = e.value.args[0]
    assert exec_msg == "Invalid input for sequence argument"

    # if the provided path doesn't contain a file
    with pytest.raises(ValueError):
        load_sequences("./tests/data/not_a_file", moltype="dna")

    # if the sequences contain characters that is not in alphabet, raise an error
    sequences = ["ACTGA", "TACGE"]
    with pytest.raises(AlphabetError):
        load_sequences(sequences, moltype="dna")


def test_global_aln():
    # parameters used for alignment tests
    s = make_dna_scoring_dict(10, -1, -8)
    d = 10
    e = 2

    # one sequence is empty tests
    seq1 = ""
    seq2 = "ACG"
    assert dna_global_aln(seq1=seq1, seq2=seq2, s=s, d=d, e=e) == ("---", "ACG")
    seq1 = "CTA"
    seq2 = ""
    assert dna_global_aln(seq1=seq1, seq2=seq2, s=s, d=d, e=e) == ("CTA", "---")

    # simple alignment test
    seq1 = "GGATCGA"
    seq2 = "GAATTCAGTTA"
    assert dna_global_aln(seq1=seq1, seq2=seq2, s=s, d=d, e=e) in [
        ("GGA-TC-G--A", "GAATTCAGTTA"),
        ("GGAT-C-G--A", "GAATTCAGTTA"),
    ]


def test_debruijn_merge_correctness():
    # Outer-most list elements don't have equal length (cycle in de Bruijn graph)
    test_lst1 = [[0, [1, 2], 3], [0, 3]]
    assert debruijn_merge_correctness(test_lst1) == False
    # Inner list elements don't have the same type (cycle in de Bruijn graph)
    test_lst2 = [[0, [1, 2], 3], [0, 4, 3]]
    assert debruijn_merge_correctness(test_lst2) == False
    # Inner list int elements don't have the same value (cycle in de Bruijn graph)
    test_lst3 = [[0, [1, 2], 3], [0, [6, 7], 4]]
    assert debruijn_merge_correctness(test_lst3) == False
    # True case, where merge node are consistent (same index), types are consistent
    test_lst4 = [[0, [1, 2], 3], [0, [6, 7], 3]]
    assert debruijn_merge_correctness(test_lst4) == True


def test_get_kmers():
    # k = 3, generating kmers test
    test_seq1 = "GTACACGTATG"
    assert get_kmers(test_seq1, 3) == [
        "GTA",
        "TAC",
        "ACA",
        "CAC",
        "ACG",
        "CGT",
        "GTA",
        "TAT",
        "ATG",
    ]

    # another k value test
    test_seq2 = "TACCACGTAAT"
    assert get_kmers(test_seq2, 4) == [
        "TACC",
        "ACCA",
        "CCAC",
        "CACG",
        "ACGT",
        "CGTA",
        "GTAA",
        "TAAT",
    ]

    # invalid k value test
    with pytest.raises(ValueError) as e:
        get_kmers(test_seq2, 16)
    exec_msg = e.value.args[0]
    assert exec_msg == "Invalid k size for kmers"


def test_duplicate_kmers():
    # find duplicate kmers test 1
    kmer_lst1 = [["AG", "CT", "CA", "AC", "CT"], ["CT", "AC", "GT", "AG"]]
    assert duplicate_kmers(kmer_lst1) == {"CT"}

    # find duplicate kmers test 2
    kmer_lst2 = [["AG", "CT", "CA", "AG", "CT"], ["CT", "AG", "CT", "AG", "AC"]]
    assert duplicate_kmers(kmer_lst2) == {"AG", "CT"}


if __name__ == "__main__":
    pytest.main()
