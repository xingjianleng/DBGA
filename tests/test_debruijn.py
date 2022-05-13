from cogent3.core.alphabet import AlphabetError
import pytest

from src.debruijn import *


def sequence_loader_checker(seqs, exp_seq1: str, exp_seq2: str):
    assert isinstance(seqs, np.ndarray)
    assert seqs[0] == exp_seq1
    assert seqs[1] == exp_seq2


# checker for check the kmer length is satisfied
def kmer_length_checker(debruijn: deBruijn):
    for node in debruijn.nodes.values():
        if node.node_type not in [NodeType.start, NodeType.end]:
            assert len(node.kmer) == debruijn.k


# fasta example checker framework
def example_checker(seqs, k, exp_seq1_idx, exp_seq2_idx, exp_merge, exp_aln):
    # expected sequence should be the same as degapped aligned sequences
    exp_seq1 = exp_aln[0][0].replace("-", "")
    exp_seq2 = exp_aln[0][1].replace("-", "")

    # all testing examples are DNA sequences
    db = deBruijn(seqs, k, moltype="dna")
    # check kmer length
    kmer_length_checker(db)
    exp_start_idx = exp_seq1_idx[0], exp_seq2_idx[0]
    exp_end_idx = exp_seq1_idx[-1], exp_seq2_idx[-1]
    # sequences should be the same
    assert db.sequences[0] == exp_seq1
    assert db.sequences[1] == exp_seq2

    # node Ids should be consistent with expected Ids
    assert db.seq_node_idx[0] == exp_seq1_idx
    assert db.seq_node_idx[1] == exp_seq2_idx
    for i in range(db.id_count):
        if i in exp_start_idx:
            assert db.nodes[i].node_type is NodeType.start
        elif i in exp_end_idx:
            assert db.nodes[i].node_type is NodeType.end
        else:
            assert db.nodes[i].node_type is NodeType.middle

    # two aligned sequences should have the same length
    aln_result = db.to_alignment()
    assert len(aln_result[0]) == len(aln_result[1])
    assert aln_result in exp_aln

    # merge node Ids should be the same as expectation
    # NOTE: this should be checked at last, because db.to_alignment() could possibly reset this list
    assert db.merge_node_idx == exp_merge


def test_read_debruijn_edge_kmer():
    # empty string test case
    seq1 = ""
    assert read_debruijn_edge_kmer(seq1, 3) == ""

    # normal case, extract first char of kmer string test
    seq2 = "AOPKFNLSDOEPKPOWSEGH"
    assert read_debruijn_edge_kmer(seq2, 4) == "AFDKS"
    seq3 = "1092831790"
    assert read_debruijn_edge_kmer(seq3, 2) == "19819"

    # length of sequence is not a multiple of k, raise error
    seq4 = "ACMDANFGL"
    with pytest.raises(AssertionError):
        read_debruijn_edge_kmer(seq4, 4)
    seq5 = "109128903"
    with pytest.raises(AssertionError):
        read_debruijn_edge_kmer(seq5, 2)


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
    # one sequence is empty tests
    seq1 = ""
    seq2 = "ACG"
    assert dna_global_aln(seq1=seq1, seq2=seq2) == ("---", "ACG")
    seq1 = "CTA"
    seq2 = ""
    assert dna_global_aln(seq1=seq1, seq2=seq2) == ("CTA", "---")

    # simple alignment test
    seq1 = "GGATCGA"
    seq2 = "GAATTCAGTTA"
    assert dna_global_aln(seq1=seq1, seq2=seq2) in [
        ("GGA-TC-G--A", "GAATTCAGTTA"),
        ("GGAT-C-G--A", "GAATTCAGTTA"),
    ]


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


def test_mapping_shift():
    # Test case 1
    seq1 = "TGTAAGCGA"
    seq2 = "GTACAAGCGA"
    db = deBruijn([seq1, seq2], k=3, moltype="dna")
    shift_res = mapping_shifts(db)
    assert shift_res == {"node_indicies": [2, 4, 5, 6, 7], "shifts": [-1, 1, 1, 1, 1]}

    # Test case 2
    seq1 = "TACCACGTAAT"
    seq2 = "TACGACCTAAT"
    db = deBruijn([seq1, seq2], k=3, moltype="dna")
    shift_res = mapping_shifts(db)
    assert shift_res == {"node_indicies": [1, 2, 5, 8, 9], "shifts": [0, 3, -3, 0, 0]}

    # Test case 3
    seq1 = "TACCGTCCAGACGTAAT"
    seq2 = "TACGGTCCAGACCTAAT"
    db = deBruijn([seq1, seq2], k=3, moltype="dna")
    shift_res = mapping_shifts(db)
    assert shift_res == {
        "node_indicies": [1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13],
        "shifts": [0, 9, 1, 1, 1, 1, 1, 1, -8, 2, 2],
    }

    # Test case 4
    db = seq1
    with pytest.raises(AssertionError):
        shift_res = mapping_shifts(db)


def test_duplicate_kmers():
    # find duplicate kmers test 1
    kmer_lst1 = [["AG", "CT", "CA", "AC", "CT"], ["CT", "AC", "GT", "AG"]]
    assert duplicate_kmers(kmer_lst1) == {"CT"}

    # find duplicate kmers test 2
    kmer_lst2 = [["AG", "CT", "CA", "AG", "CT"], ["CT", "AG", "CT", "AG", "AC"]]
    assert duplicate_kmers(kmer_lst2) == {"AG", "CT"}


def test_substitution_middle():
    # NOTE: Basic substitution test
    seq1 = "GTACAAGCGA"
    seq2 = "GTACACGCGA"
    example_checker(
        seqs="./tests/data/substitution_middle.fasta",
        k=3,
        exp_seq1_idx=list(range(10)),
        exp_seq2_idx=[10, 1, 2, 3, 11, 12, 13, 7, 8, 14],
        exp_merge=[1, 2, 3, 7, 8],
        exp_aln=[(seq1, seq2)],
    )


def test_gap_bubble_duplicate():
    # NOTE: Bubble in the middle (gap) test, with duplicate kmers
    seq1 = "GTACACGTATG"
    seq2 = "GTACACG-ATG"
    example_checker(
        seqs="./tests/data/gap_bubble_duplicate.fasta",
        k=3,
        exp_seq1_idx=list(range(9)),
        exp_seq2_idx=[9, 1, 2, 3, 4, 10, 11, 7, 12],
        exp_merge=[1, 2, 3, 4, 7],
        exp_aln=[(seq1, seq2)],
    )


def test_gap_at_end():
    # NOTE: Gaps at unbalanced ends of sequences
    seq1 = "GTACAAGCGATG"
    seq2 = "GTACAAGCGA--"
    example_checker(
        seqs="./tests/data/gap_at_end.fasta",
        k=3,
        exp_seq1_idx=list(range(12)),
        exp_seq2_idx=[12, 1, 2, 3, 4, 5, 6, 7, 8, 13],
        exp_merge=[1, 2, 3, 4, 5, 6, 7, 8],
        exp_aln=[(seq1, seq2)],
    )


def test_gap_at_start():
    # NOTE: Gaps at unbalanced starts of sequences
    seq1 = "TGTACAAGCGA"
    seq2 = "-GTACAAGCGA"
    example_checker(
        seqs="./tests/data/gap_at_start.fasta",
        k=3,
        exp_seq1_idx=list(range(11)),
        exp_seq2_idx=[11, 2, 3, 4, 5, 6, 7, 8, 9, 12],
        exp_merge=[2, 3, 4, 5, 6, 7, 8, 9],
        exp_aln=[(seq1, seq2)],
    )


def test_bubble_consecutive_duplicate():
    # NOTE: Subtitution in the middle. Two duplicate kmers (consecutive)
    seq1 = "TGTACTGTAGA"
    seq2 = "TGTACTATAGA"
    example_checker(
        seqs="./tests/data/bubble_consecutive_duplicate.fasta",
        k=3,
        exp_seq1_idx=list(range(7)),
        exp_seq2_idx=[7, 1, 2, 8, 9, 10, 4, 5, 11],
        exp_merge=[1, 2, 4, 5],
        exp_aln=[(seq1, seq2)],
    )


def test_duplicate_unbalanced_end():
    # NOTE: Very comprehensive test, duplicate kmers (anywhere, even after merge node), unbalanced ends
    seq1 = "TGTACGTCAATGTCG"
    seq2 = "TGTAAGTCAATG---"
    example_checker(
        seqs="./tests/data/duplicate_unbalanced_end.fasta",
        k=3,
        exp_seq1_idx=list(range(11)),
        exp_seq2_idx=[11, 1, 12, 13, 14, 5, 6, 7, 8, 15],
        exp_merge=[1, 5, 6, 7, 8],
        exp_aln=[(seq1, seq2)],
    )


def test_unbalanced_end_w_duplicate():
    # NOTE: Very comprehensive test, duplicate kmers (anywhere, even after merge node), unbalanced ends
    #       Sequence end by duplicate kmer
    seq1 = "TGTACGTCAATGTCG"
    seq2 = "TGTAAGTCAATGT--"
    example_checker(
        seqs="./tests/data/unbalanced_end_w_duplicate.fasta",
        k=3,
        exp_seq1_idx=list(range(11)),
        exp_seq2_idx=[11, 1, 12, 13, 14, 5, 6, 7, 8, 15],
        exp_merge=[1, 5, 6, 7, 8],
        exp_aln=[(seq1, seq2)],
    )


def test_duplicate_kmer_in_bubble():
    # NOTE: There are many duplicate kmers in this test case.
    #       They are all in the bubble of the de Bruijn graph
    seq1 = "TAC-ACGTAAT"
    seq2 = "TACGACG-AAT"
    example_checker(
        seqs="./tests/data/duplicate_kmer_in_bubble.fasta",
        k=3,
        exp_seq1_idx=list(range(9)),
        exp_seq2_idx=[9, 1, 10, 11, 7, 12],
        exp_merge=[1, 7],
        exp_aln=[(seq1, seq2)],
    )


def test_both_end_duplicate():
    # NOTE: Two sequences end with duplicate kmers, and these duplicate kmers are connected
    #       to a merge node.
    seq1 = "ACGTGACG"
    seq2 = "ACTTGACG"
    example_checker(
        seqs="./tests/data/both_end_duplicate.fasta",
        k=3,
        exp_seq1_idx=list(range(6)),
        exp_seq2_idx=[6, 7, 8, 9, 3, 4, 10],
        exp_merge=[3, 4],
        exp_aln=[(seq1, seq2)],
    )


def test_unrelated_sequences():
    # NOTE: Two sequences do not share any common kmer
    seq1 = "ACGT--AGTATC"
    seq2 = "ACCTACAGCA--"
    example_checker(
        seqs="./tests/data/unrelated_sequences.fasta",
        k=3,
        exp_seq1_idx=list(range(8)),
        exp_seq2_idx=list(range(8, 18)),
        exp_merge=[],
        exp_aln=[(seq1, seq2)],
    )


if __name__ == "__main__":
    pytest.main()
