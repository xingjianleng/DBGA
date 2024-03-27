import os
from cogent3 import load_unaligned_seqs
from cogent3.align import make_dna_scoring_dict
from dbga.adaptive_debruijn import adpt_dbg_alignment_recursive
import pytest


def adaptive_dbg_alignment_checker(file, k_choice, aln_str_res):
    # helper function to check the recursive part of adaptive de Bruijn alignment
    s = make_dna_scoring_dict(10, -1, -8)
    input = [str(seq) for seq in 
        load_unaligned_seqs(file,moltype="dna").seqs
    ]
    
    assert adpt_dbg_alignment_recursive(
        input, k_index=0, k_choice=k_choice, s=s, d=10, e=2) == aln_str_res


def test_adpt_dbg_alignment_recursive_1(data_dir):
    # simple test on adaptive de Bruijn alignment
    file = os.path.join(data_dir,"gap_bubble_duplicate.fasta")
    k_choice = (3, 2)
    aln_str_res = ("GTACACGTATG", "GTACACG-ATG")
    adaptive_dbg_alignment_checker(file, k_choice, aln_str_res)


def test_adpt_dbg_alignment_recursive_2(data_dir):
    # test case to see whether the algorithm can filter invalid k choices
    file = os.path.join(data_dir,"gap_bubble_duplicate.fasta")
    k_choice = (7, 5, 3, 2)
    aln_str_res = ("GTACACGTATG", "GTACACG-ATG")
    adaptive_dbg_alignment_checker(file, k_choice, aln_str_res)


def test_adpt_dbg_alignment_recursive_3(data_dir):
    # test case when restricting 
    file = os.path.join(data_dir,"duplicate_unbalanced_end.fasta")
    k_choice = (3,)
    aln_str_res = ("TGTACGTCAATGTCG", "TGTAAGTCAATG---")
    adaptive_dbg_alignment_checker(file, k_choice, aln_str_res)


def test_adpt_dbg_alignment_recursive_4(data_dir):
    # test case where the duplicate k-mers exists in the stem
    file = os.path.join(data_dir,"duplicate_in_stem.fasta")
    k_choice = (3,)
    aln_str_res = ("GTACACGTATG", "GTACTCGTATG")
    adaptive_dbg_alignment_checker(file, k_choice ,aln_str_res)


if __name__ == "__main__":
    pytest.main()
