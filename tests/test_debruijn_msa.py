from dbga.debruijn_msa import *
import pytest


# checker for check the kmer length is satisfied
def kmer_length_checker(debruijn: deBruijnMultiSeqs):
    for node in debruijn.nodes.values():
        if node.node_type not in [NodeType.start, NodeType.end]:
            assert len(node.kmer) == debruijn.k


def msa_three_seqs_checker(seqs, k, exp_seqs_idx, exp_merge, exp_aln):
    exp_seqs = [exp_seq.replace("-", "") for exp_seq in exp_aln]
    # all testing examples are DNA sequences
    db = deBruijnMultiSeqs(seqs, k, moltype="dna")
    # check kmer length
    kmer_length_checker(db)

    # sequences should be the same
    for i, seqs in enumerate(db.sequences):
        assert seqs == exp_seqs[i]

    # node Ids should be consistent with expected Ids
    for i in range(db.num_seq):
        assert db.seq_node_idx[i] == exp_seqs_idx[i]

    # two aligned sequences should have the same length
    aln_result = db.alignment().to_dict()

    assert aln_result == {f"seq{i + 1}": exp_aln[i] for i in range(db.num_seq)}

    # merge node IDs should be the same as expectation
    assert db.merge_node_idx == exp_merge


def test_msa_example1():
    seq1 = "ACTTGACAGCT"
    seq2 = "ACATGACAGCT"
    seq3 = "ACTTGACACGT"
    msa_three_seqs_checker(
        seqs="data/msa_example1.fasta",
        k=3,
        exp_seqs_idx=[
            list(range(10)),
            [10, 11, 12, 4, 5, 6, 7, 8, 13],
            [14, 1, 2, 3, 4, 5, 15, 16, 17, 18],
        ],
        exp_merge=[4, 5],
        exp_aln=(seq1, seq2, seq3),
    )


def test_msa_example2_not_optimal():
    seq1 = "GTAATTGCCACGCGA--"
    seq2 = "GTAATTGCCT--CGAGA"
    seq3 = "GTAATTGCCACGCGA--"
    msa_three_seqs_checker(
        seqs="data/msa_example2.fasta",
        k=3,
        exp_seqs_idx=[
            list(range(15)),
            [15, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 13, 19, 20, 21],
            [22, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 23],
        ],
        exp_merge=[1, 2, 3, 4, 5, 6, 7, 13],
        exp_aln=(seq1, seq2, seq3),
    )


def test_msa_example2_success():
    seq1 = "GTAATTGCCACGCGA"
    seq2 = "GTAATTGCCTCGAGA"
    seq3 = "GTAATTGCCACGCGA"
    msa_three_seqs_checker(
        seqs="data/msa_example2.fasta",
        k=4,
        exp_seqs_idx=[
            list(range(14)),
            [14, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20, 21],
            [22, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 23],
        ],
        exp_merge=list(range(1, 7)),
        exp_aln=(seq1, seq2, seq3),
    )


def test_msa_example3_fail():
    with pytest.raises(ValueError) as e:
        deBruijnMultiSeqs("data/msa_example3.fasta", k=3, moltype="dna")
        assert (
            e.value
            == "Cycles detected in de Bruijn graph, usually caused by small kmer sizes"
        )


def test_msa_example3_success():
    seq1 = "TACCGTCCAGACGTAAT"
    seq2 = "TACGGTCCAGACCTAAT"
    seq3 = "TACGGTCCAGACCTAAT"
    msa_three_seqs_checker(
        seqs="data/msa_example3.fasta",
        k=4,
        exp_seqs_idx=[
            list(range(16)),
            [16, 17, 18, 19, 20, 5, 6, 7, 8, 9, 21, 22, 23, 24, 14, 25],
            [26, 17, 18, 19, 20, 5, 6, 7, 8, 9, 21, 22, 23, 24, 14, 27],
        ],
        exp_merge=[5, 6, 7, 8, 9, 14],
        exp_aln=(seq1, seq2, seq3),
    )


def test_msa_example4():
    seq1 = "GTACAAGCGATG"
    seq2 = "GTACAAGCGA--"
    seq3 = "GTAC-AGCGATG"
    msa_three_seqs_checker(
        seqs="data/msa_example4.fasta",
        k=3,
        exp_seqs_idx=[
            list(range(12)),
            [12, 1, 2, 3, 4, 5, 6, 7, 8, 13],
            [14, 1, 2, 3, 15, 6, 7, 8, 9, 10, 16],
        ],
        exp_merge=[1, 2, 3, 6, 7, 8],
        exp_aln=(seq1, seq2, seq3),
    )


if __name__ == "__main__":
    pytest.main()
