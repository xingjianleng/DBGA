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
def example_checker(
        path,
        k,
        exp_seq1,
        exp_seq2,
        exp_seq1_idx,
        exp_seq2_idx,
        exp_merge,
        exp_aln):
    db = deBruijn(path, k)
    # check kmer length
    kmer_length_checker(db)
    exp_start_idx = [indices[0] for indices in [exp_seq1_idx, exp_seq2_idx]]
    exp_end_idx = [indices[-1] for indices in [exp_seq1_idx, exp_seq2_idx]]
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
    aln_result = db.to_Alignment()
    assert len(aln_result[0]) == len(aln_result[1])
    assert aln_result in exp_aln

    # merge node Ids should be the same as expectation
    # NOTE: this should be checked at last, because db.to_Alignment() could possibly reset this list
    assert db.merge_node_idx == exp_merge


def test_balancing_aln_seqs():
    # equal length test
    seq1 = 'AGCTCTAGT'
    seq2 = 'TAGACTGTA'
    balancing_res = balancing_aln_seqs(seq1, seq2)
    assert balancing_res == (seq1, seq2)

    # first sequence balancing test
    seq3 = 'AGCTCTA----'
    balancing_res = balancing_aln_seqs(seq3, seq2)
    assert balancing_res == ('AGCTCTA--', seq2)

    # second sequence balancing test
    seq4 = 'TACTGTA----'
    balancing_res = balancing_aln_seqs(seq1, seq4)
    assert balancing_res == (seq1, 'TACTGTA--')

    # Error test, incorrect alignment in sequence 1
    seq5 = 'AGCTCTAGTAG'
    seq6 = 'TAGACTGTA-'
    with pytest.raises(ValueError) as e:
        balancing_res = balancing_aln_seqs(seq5, seq6)
    exec_msg = e.value.args[0]
    assert exec_msg == 'Input sequences are not appropriately aligned in sequence 1!'

    # Error test, incorrect alignment in sequence 2
    seq7 = 'AGCTCTAGT--'
    seq8 = 'TAGACTGTATCG'
    with pytest.raises(ValueError) as e:
        balancing_res = balancing_aln_seqs(seq7, seq8)
    exec_msg = e.value.args[0]
    assert exec_msg == 'Input sequences are not appropriately aligned in sequence 2!'


def test_read_debruijn_edge_kmer():
    # empty string test case
    seq1 = ''
    assert read_debruijn_edge_kmer(seq1, 3) == ''

    # normal case, extract first char of kmer string test
    seq2 = 'AOPKFNLSDOEPKPOWSEGH'
    assert read_debruijn_edge_kmer(seq2, 4) == 'AFDKS'
    seq3 = '1092831790'
    assert read_debruijn_edge_kmer(seq3, 2) == '19819'

    # length of sequence is not a multiple of k, raise error
    seq4 = 'ACMDANFGL'
    with pytest.raises(AssertionError):
        read_debruijn_edge_kmer(seq4, 4)
    seq5 = '109128903'
    with pytest.raises(AssertionError):
        read_debruijn_edge_kmer(seq5, 2)


def test_load_sequences():
    # given a path, load sequences
    path = './tests/data/ex1.fasta'
    sequence_loader_checker(load_sequences(path), 'GTACAAGCGA', 'GTACACGCGA')

    # given a SequenceCollection, load sequences
    seq1 = 'ACGTCAG'
    seq2 = 'GACTGTGA'
    sc = SequenceCollection({'seq1': seq1, 'seq2': seq2})
    sequence_loader_checker(load_sequences(sc), seq1, seq2)

    # given a list of strings, load sequences
    seq_lst = ['ACGTCAG', 'GACTGTGA']
    sequence_loader_checker(load_sequences(seq_lst), seq_lst[0], seq_lst[1])

    # if the provided parameter is in a wrong type, raise error
    with pytest.raises(TypeError) as e:
        load_sequences({1: 'ACGTGTA'})
    exec_msg = e.value.args[0]
    assert exec_msg == 'Invalid input type for sequence argument'

    # if the provided path doesn't contain a file
    with pytest.raises(ValueError):
        load_sequences('./tests/data/not_a_file')


def test_lcs():
    # find lcs of list of char test
    lst1 = ['B', 'C', 'D', 'A', 'A', 'C', 'D']
    lst2 = ['A', 'C', 'D', 'B', 'A', 'C']
    lcs_result = lcs(lst1, lst2)
    assert lcs_result == ['C', 'D', 'A', 'C']

    # find lcs of list of int test
    lst1 = [3, 6, 1, 8, 10]
    lst2 = [1, 8, 6]
    lcs_result = lcs(lst1, lst2)
    assert lcs_result == [1, 8]


def test_global_aln():
    # one sequence is empty tests
    seq1 = ''
    seq2 = 'ACG'
    assert global_aln(seq1=seq1, seq2=seq2) == ('---', 'ACG')
    seq1 = 'CTA'
    seq2 = ''
    assert global_aln(seq1=seq1, seq2=seq2) == ('CTA', '---')

    # simple alignment test
    seq1 = 'GGATCGA'
    seq2 = 'GAATTCAGTTA'
    assert global_aln(seq1=seq1, seq2=seq2) in [
        ('GGA-TC-G--A', 'GAATTCAGTTA'),
        ('GGAT-C-G--A', 'GAATTCAGTTA')
    ]


def test_get_kmers():
    # k = 3, generating kmers test
    test_seq1 = 'GTACACGTATG'
    assert get_kmers(test_seq1, 3) == [
        'GTA',
        'TAC',
        'ACA',
        'CAC',
        'ACG',
        'CGT',
        'GTA',
        'TAT',
        'ATG'
    ]

    # another k value test
    test_seq2 = 'TACCACGTAAT'
    assert get_kmers(test_seq2, 4) == [
        'TACC',
        'ACCA',
        'CCAC',
        'CACG',
        'ACGT',
        'CGTA',
        'GTAA',
        'TAAT'
    ]

    # invalid k value test
    with pytest.raises(ValueError) as e:
        get_kmers(test_seq2, 16)
    exec_msg = e.value.args[0]
    assert exec_msg == 'Invalid k size for kmers'


def test_duplicate_kmers():
    # find duplicate kmers test 1
    kmer_lst1 = [
        ['AG', 'CT', 'CA', 'AC', 'CT'],
        ['CT', 'AC', 'GT', 'AG']
    ]
    assert duplicate_kmers(kmer_lst1) == {'CT'}

    # find duplicate kmers test 2
    kmer_lst2 = [
        ['AG', 'CT', 'CA', 'AG', 'CT'],
        ['CT', 'AG', 'CT', 'AG', 'AC']
    ]
    assert duplicate_kmers(kmer_lst2) == {'AG', 'CT'}


def test_ex1():
    # NOTE: Basic substitution test
    # seq1: GTACAAGCGA
    # seq2: GTACACGCGA
    example_checker(
        path='./tests/data/ex1.fasta',
        k=3,
        exp_seq1='GTACAAGCGA',
        exp_seq2='GTACACGCGA',
        exp_seq1_idx=list(range(10)),
        exp_seq2_idx=[10, 1, 2, 3, 11, 12, 13, 7, 8, 14],
        exp_merge=[1, 2, 3, 7, 8],
        exp_aln=[('GTACAAGCGA', 'GTACACGCGA')]
    )


def test_ex2():
    # NOTE: Bubble in the middle (gap) test, with duplicate kmers
    # seq1: GTACACGTATG
    # seq2: GTACACGATG
    example_checker(
        path='./tests/data/ex2.fasta',
        k=3,
        exp_seq1='GTACACGTATG',
        exp_seq2='GTACACGATG',
        exp_seq1_idx=list(range(9)),
        exp_seq2_idx=[9, 1, 2, 3, 4, 10, 11, 7, 12],
        exp_merge=[1, 2, 3, 4, 7],
        exp_aln=[("GTACACGTATG", "GTACACG-ATG")]
    )


def test_ex3():
    # NOTE: Gaps at unbalanced ends of sequences
    # seq1: GTACAAGCGATG
    # seq2: GTACAAGCGA
    example_checker(
        path='./tests/data/ex3.fasta',
        k=3,
        exp_seq1='GTACAAGCGATG',
        exp_seq2='GTACAAGCGA',
        exp_seq1_idx=list(range(12)),
        exp_seq2_idx=[
            12, 1, 2, 3, 4, 5, 6, 7, 8, 13
        ],
        exp_merge=[1, 2, 3, 4, 5, 6, 7, 8],
        exp_aln=[("GTACAAGCGATG", "GTACAAGCGA--")]
    )


def test_ex4():
    # NOTE: Gaps at unbalanced starts of sequences
    # seq1: TGTACAAGCGA
    # seq2: GTACAAGCGA
    example_checker(
        path='./tests/data/ex4.fasta',
        k=3,
        exp_seq1='TGTACAAGCGA',
        exp_seq2='GTACAAGCGA',
        exp_seq1_idx=list(range(11)),
        exp_seq2_idx=[
            11, 2, 3, 4, 5, 6, 7, 8, 9, 12
        ],
        exp_merge=[2, 3, 4, 5, 6, 7, 8, 9],
        exp_aln=[("TGTACAAGCGA", "-GTACAAGCGA")]
    )


def test_ex5():
    # NOTE: Subtitution in the middle. Two duplicate kmers (consecutive)
    # seq1: TGTACTGTAGA
    # seq2: TGTACTATAGA
    example_checker(
        path='./tests/data/ex5.fasta',
        k=3,
        exp_seq1='TGTACTGTAGA',
        exp_seq2='TGTACTATAGA',
        exp_seq1_idx=list(range(7)),
        exp_seq2_idx=[
            7, 1, 2, 8, 9, 10, 4, 5, 11
        ],
        exp_merge=[1, 2, 4, 5],
        exp_aln=[("TGTACTGTAGA", "TGTACTATAGA")]
    )


def test_ex6():
    # NOTE: Very comprehensive test, duplicate kmers (anywhere, even after merge node), unbalanced ends
    # seq1: TGTACGTCAATGTCG
    # seq2: TGTAAGTCAATG
    example_checker(
        path='./tests/data/ex6.fasta',
        k=3,
        exp_seq1='TGTACGTCAATGTCG',
        exp_seq2='TGTAAGTCAATG',
        exp_seq1_idx=list(range(11)),
        exp_seq2_idx=[
            11, 1, 12, 13, 14, 5, 6, 7, 8, 15
        ],
        exp_merge=[1, 5, 6, 7, 8],
        exp_aln=[("TGTACGTCAATGTCG", "TGTAAGTCAATG---")]
    )


@pytest.mark.xfail(reason="Algorithm can't tackle with this edge case!")
def test_edge_case1():
    # NOTE: Duplicate kmer at different position of different sequences lead to cycles
    #       No merged kmers between cycles (This case LCS doesn't help)
    # seq1: TACCACGTAAT
    # seq2: TACGACCTAAT
    example_checker(
        path='./tests/data/edge_case1.fasta',
        k=3,
        exp_seq1='TACCACGTAAT',
        exp_seq2='TACGACCTAAT',
        exp_seq1_idx=list(range(11)),
        exp_seq2_idx=[11, 1, 5, 12, 13, 2, 14, 15, 8, 9, 16],
        exp_merge=[1, 8, 9],
        exp_aln=[("TACCACGTAAT", "TACGACCTAAT")]
    )


def test_edge_case2():
    # NOTE: Duplicate kmer at different position of different sequences lead to cycles
    #       With merged kmers between cycles (This case we can use LCS to solve the problem)
    # seq1: TACCGTCCAGACGTAAT
    # seq2: TACGGTCCAGACCTAAT
    example_checker(
        path='./tests/data/edge_case2.fasta',
        k=3,
        exp_seq1='TACCGTCCAGACGTAAT',
        exp_seq2='TACGGTCCAGACCTAAT',
        exp_seq1_idx=list(range(15)),
        exp_seq2_idx=[
            15, 1, 10, 16, 17, 4, 5, 6, 7, 8, 9, 2, 18, 19, 12, 13, 20
        ],
        exp_merge=[1, 4, 5, 6, 7, 8, 9, 12, 13],
        exp_aln=[(
            "TACCGTCCAGACGTAAT", "TACGGTCCAGACCTAAT"
        )]
    )


if __name__ == '__main__':
    pytest.main()
