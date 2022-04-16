import pytest

from src.debruijn import NodeType, deBruijn, duplicate_kmers, get_kmers, global_aln, lcs


def kmer_length_checker(debruijn: deBruijn):
    for node in debruijn.nodes.values():
        if node.node_type not in [NodeType.start, NodeType.end]:
            assert len(node.kmer) == debruijn.k


def test_lcs():
    lst1 = ['B', 'C', 'D', 'A', 'A', 'C', 'D']
    lst2 = ['A', 'C', 'D', 'B', 'A', 'C']
    lcs_result = lcs(lst1, lst2)
    assert lcs_result == ['C', 'D', 'A', 'C']


def test_global_aln():
    seq1 = ''
    seq2 = 'ACG'
    assert global_aln(seq1=seq1, seq2=seq2) == ('---', 'ACG')
    seq1 = 'CTA'
    seq2 = ''
    assert global_aln(seq1=seq1, seq2=seq2) == ('CTA', '---')
    seq1 = 'GGATCGA'
    seq2 = 'GAATTCAGTTA'
    assert global_aln(seq1=seq1, seq2=seq2) in [
        ('GGA-TC-G--A', 'GAATTCAGTTA'),
        ('GGAT-C-G--A', 'GAATTCAGTTA')
    ]


def test_get_kmers():
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
    with pytest.raises(ValueError) as e:
        get_kmers(test_seq2, 16)
    exec_msg = e.value.args[0]
    assert exec_msg == 'Invalid k size for kmers'


def test_duplicate_kmers():
    kmer_lst1 = [
        ['AG', 'CT', 'CA', 'AC', 'CT'],
        ['CT', 'AC', 'GT', 'AG']
    ]
    assert duplicate_kmers(kmer_lst1) == {'CT'}
    kmer_lst2 = [
        ['AG', 'CT', 'CA', 'AG', 'CT'],
        ['CT', 'AG', 'CT', 'AG', 'AC']
    ]
    assert duplicate_kmers(kmer_lst2) == {'AG', 'CT'}


def test_kmer_length():
    for i in range(1, 6):
        path = f'./tests/data/ex{i}.fasta'
        d = deBruijn(path, 3)
        kmer_length_checker(d)


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
    exp_start_idx = [indices[0] for indices in [exp_seq1_idx, exp_seq2_idx]]
    exp_end_idx = [indices[-1] for indices in [exp_seq1_idx, exp_seq2_idx]]
    assert db.sequences[0] == exp_seq1
    assert db.sequences[1] == exp_seq2
    assert db.seq_node_idx[0] == exp_seq1_idx
    assert db.seq_node_idx[1] == exp_seq2_idx
    assert db.merge_node_idx == exp_merge
    for i in range(db.id_count):
        if i in exp_start_idx:
            assert db.nodes[i].node_type is NodeType.start
        elif i in exp_end_idx:
            assert db.nodes[i].node_type is NodeType.end
        else:
            assert db.nodes[i].node_type is NodeType.middle
    aln_result = db.to_Alignment()
    assert aln_result in exp_aln


def test_ex1():
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
    # seq1: GTACACGCTA
    # seq2: GTACACGCGA
    example_checker(
        path='./tests/data/ex2.fasta',
        k=3,
        exp_seq1='GTACACGCTA',
        exp_seq2='GTACACGCGA',
        exp_seq1_idx=list(range(10)),
        exp_seq2_idx=[10, 1, 2, 3, 4, 5, 6, 11, 12, 13],
        exp_merge=[1, 2, 3, 4, 5, 6],
        exp_aln=[('GTACACGCTA', 'GTACACGCGA')]
    )


def test_ex3():
    # seq1: GTACACGTATG
    # seq2: GTACACGATG
    example_checker(
        path='./tests/data/ex3.fasta',
        k=3,
        exp_seq1='GTACACGTATG',
        exp_seq2='GTACACGATG',
        exp_seq1_idx=list(range(9)),
        exp_seq2_idx=[9, 1, 2, 3, 4, 10, 11, 7, 12],
        exp_merge=[1, 2, 3, 4, 7],
        exp_aln=[("GTACACGTATG", "GTACACG-ATG")]
    )


def test_ex4():
    # seq1: GTACAAGCGA
    # seq2: GTACAGCGA
    example_checker(
        path='./tests/data/ex4.fasta',
        k=3,
        exp_seq1='GTACAAGCGA',
        exp_seq2='GTACAGCGA',
        exp_seq1_idx=list(range(10)),
        exp_seq2_idx=[10, 1, 2, 3, 11, 6, 7, 8, 12],
        exp_merge=[1, 2, 3, 6, 7, 8],
        exp_aln=[
            ("GTACAAGCGA", "GTACA-GCGA"), ("GTACAAGCGA", "GTAC-AGCGA")
        ]
    )


def test_ex5():
    # seq1: ACGTAGACG
    # seq2: ACGTAACG
    example_checker(
        path='./tests/data/ex5.fasta',
        k=3,
        exp_seq1='ACGTAGACG',
        exp_seq2='ACGTAACG',
        exp_seq1_idx=list(range(7)),
        exp_seq2_idx=[7, 1, 2, 8, 9, 10],
        exp_merge=[1, 2],
        exp_aln=[("ACGTAGACG", "ACGTA-ACG")]
    )


if __name__ == '__main__':
    pytest.main()
