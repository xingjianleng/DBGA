from dbga.debruijn_pairwise import *
import pytest


# checker for check the kmer length is satisfied
def kmer_length_checker(debruijn: DeBruijnPairwise):
    for node in debruijn.nodes.values():
        if node.node_type not in [NodeType.start, NodeType.end]:
            assert len(node.kmer) == debruijn.k


# fasta example checker framework
def example_checker(seqs, k, exp_seq1_idx, exp_seq2_idx, exp_merge, exp_aln):
    # expected sequence should be the same as degapped aligned sequences
    exp_seq1 = exp_aln[0].replace("-", "")
    exp_seq2 = exp_aln[1].replace("-", "")

    # all testing examples are DNA sequences
    db = DeBruijnPairwise(seqs, k, moltype="dna")
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
    aln_result = db.alignment()

    assert aln_result == {f"seq{i + 1}": exp_aln[i] for i in range(db.num_seq)}

    # merge node IDs should be the same as expectation
    # NOTE: this should be checked at last, because to_alignment(db) could possibly reset this list
    assert db.merge_node_idx == exp_merge


def test_substitution_middle():
    # NOTE: Basic substitution test
    seq1 = "GTACAAGCGA"
    seq2 = "GTACACGCGA"
    example_checker(
        seqs="data/substitution_middle.fasta",
        k=3,
        exp_seq1_idx=list(range(10)),
        exp_seq2_idx=[10, 1, 2, 3, 11, 12, 13, 7, 8, 14],
        exp_merge=[1, 2, 3, 7, 8],
        exp_aln=(seq1, seq2),
    )


def test_gap_bubble_duplicate():
    # NOTE: Bubble in the middle (gap) test, with duplicate kmers
    seq1 = "GTACACGTATG"
    seq2 = "GTACACG-ATG"
    example_checker(
        seqs="data/gap_bubble_duplicate.fasta",
        k=3,
        exp_seq1_idx=list(range(9)),
        exp_seq2_idx=[9, 1, 2, 3, 4, 10, 11, 7, 12],
        exp_merge=[1, 2, 3, 4, 7],
        exp_aln=(seq1, seq2),
    )


def test_gap_at_end():
    # NOTE: Gaps at unbalanced ends of sequences
    seq1 = "GTACAAGCGATG"
    seq2 = "GTACAAGCGA--"
    example_checker(
        seqs="data/gap_at_end.fasta",
        k=3,
        exp_seq1_idx=list(range(12)),
        exp_seq2_idx=[12, 1, 2, 3, 4, 5, 6, 7, 8, 13],
        exp_merge=[1, 2, 3, 4, 5, 6, 7, 8],
        exp_aln=(seq1, seq2),
    )


def test_gap_at_start():
    # NOTE: Gaps at unbalanced starts of sequences
    seq1 = "TGTACAAGCGA"
    seq2 = "-GTACAAGCGA"
    example_checker(
        seqs="data/gap_at_start.fasta",
        k=3,
        exp_seq1_idx=list(range(11)),
        exp_seq2_idx=[11, 2, 3, 4, 5, 6, 7, 8, 9, 12],
        exp_merge=[2, 3, 4, 5, 6, 7, 8, 9],
        exp_aln=(seq1, seq2),
    )


def test_bubble_consecutive_duplicate():
    # NOTE: Subtitution in the middle. Two duplicate kmers (consecutive)
    seq1 = "TGTACTGTAGA"
    seq2 = "TGTACTATAGA"
    example_checker(
        seqs="data/bubble_consecutive_duplicate.fasta",
        k=3,
        exp_seq1_idx=list(range(7)),
        exp_seq2_idx=[7, 1, 2, 8, 9, 10, 4, 5, 11],
        exp_merge=[1, 2, 4, 5],
        exp_aln=(seq1, seq2),
    )


def test_duplicate_unbalanced_end():
    # NOTE: Very comprehensive test, duplicate kmers (anywhere, even after merge node), unbalanced ends
    seq1 = "TGTACGTCAATGTCG"
    seq2 = "TGTAAGTCAATG---"
    example_checker(
        seqs="data/duplicate_unbalanced_end.fasta",
        k=3,
        exp_seq1_idx=list(range(11)),
        exp_seq2_idx=[11, 1, 12, 13, 14, 5, 6, 7, 8, 15],
        exp_merge=[1, 5, 6, 7, 8],
        exp_aln=(seq1, seq2),
    )


def test_unbalanced_end_w_duplicate():
    # NOTE: Very comprehensive test, duplicate kmers (anywhere, even after merge node), unbalanced ends
    #       Sequence end by duplicate kmer
    seq1 = "TGTACGTCAATGTCG"
    seq2 = "TGTAAGTCAATGT--"
    example_checker(
        seqs="data/unbalanced_end_w_duplicate.fasta",
        k=3,
        exp_seq1_idx=list(range(11)),
        exp_seq2_idx=[11, 1, 12, 13, 14, 5, 6, 7, 8, 15],
        exp_merge=[1, 5, 6, 7, 8],
        exp_aln=(seq1, seq2),
    )


def test_duplicate_kmer_in_bubble():
    # NOTE: There are many duplicate kmers in this test case.
    #       They are all in the bubble of the de Bruijn graph
    seq1 = "TAC-ACGTAAT"
    seq2 = "TACGACG-AAT"
    example_checker(
        seqs="data/duplicate_kmer_in_bubble.fasta",
        k=3,
        exp_seq1_idx=list(range(9)),
        exp_seq2_idx=[9, 1, 10, 11, 7, 12],
        exp_merge=[1, 7],
        exp_aln=(seq1, seq2),
    )


def test_both_end_duplicate():
    # NOTE: Two sequences end with duplicate kmers, and these duplicate kmers are connected
    #       to a merge node.
    seq1 = "ACGTGACG"
    seq2 = "ACTTGACG"
    example_checker(
        seqs="data/both_end_duplicate.fasta",
        k=3,
        exp_seq1_idx=list(range(6)),
        exp_seq2_idx=[6, 7, 8, 9, 3, 4, 10],
        exp_merge=[3, 4],
        exp_aln=(seq1, seq2),
    )


def test_unrelated_sequences():
    # NOTE: Two sequences do not share any common kmer
    seq1 = "ACGT--AGTATC"
    seq2 = "ACCTACAGCA--"
    example_checker(
        seqs="data/unrelated_sequences.fasta",
        k=3,
        exp_seq1_idx=list(range(8)),
        exp_seq2_idx=list(range(8, 18)),
        exp_merge=[],
        exp_aln=(seq1, seq2),
    )


def test_edge_case1():
    # NOTE: Two sequences whose de Bruijn graph contain a circle, expected to throw an error
    seq1 = "TACCACGTAAT"
    seq2 = "TACGACCTAAT"
    with pytest.raises(ValueError) as e:
        DeBruijnPairwise([seq1, seq2], k=3, moltype="dna")
        assert (
            e.value
            == "Cycles detected in de Bruijn graph, usually caused by small kmer sizes"
        )


def test_edge_case2():
    # NOTE: Two sequences whose de Bruijn graph contain a more complex circle, expected to throw an error
    seq1 = "TACCGTCCAGACGTAAT"
    seq2 = "TACGGTCCAGACCTAAT"
    with pytest.raises(ValueError) as e:
        DeBruijnPairwise([seq1, seq2], k=3, moltype="dna")
        assert (
            e.value
            == "Cycles detected in de Bruijn graph, usually caused by small kmer sizes"
        )


if __name__ == "__main__":
    pytest.main()
