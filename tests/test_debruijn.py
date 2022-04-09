import unittest

from src.debruijn import deBruijn


class DeBruijnTest(unittest.TestCase):

    def assertSetListEquals(self, input_set, input_list):
        self.assertEqual(len(input_set), len(input_list))
        for elem in input_list:
            self.assertIn(elem, input_set)

    def _check_kmer_length(self, debruijn: deBruijn):
        for node in debruijn.nodes.values():
            if node.kmer not in ['#', '$']:
                self.assertEqual(len(node.kmer), debruijn.k)

    def _debruijn_w_basic_check(self, arg0):
        result = deBruijn(arg0, 3)
        self._check_kmer_length(result)
        self.assertEqual(result.num_seq, 2)
        return result

    def test_ex1(self):
        d = self._debruijn_w_basic_check('./tests/data/ex1.fasta')
        self.assertListEqual(
            d.seq_node_idx[0], list(range(10))
        )
        self.assertListEqual(
            d.seq_node_idx[1], [10, 1, 2, 3, 11, 12, 13, 7, 8, 14]
        )
        self.assertListEqual(d.merge_node_idx, [1, 2, 3, 7, 8])
        self.assertListEqual(d.seq_end_idx, [8, 8])
        # only substitution
        self.assertTupleEqual(d.to_POA(), ("GTACAAGCGA", "GTACACGCGA"))

    def test_ex2(self):
        d = self._debruijn_w_basic_check('./tests/data/ex2.fasta')
        self.assertListEqual(
            d.seq_node_idx[0], list(range(10))
        )
        self.assertListEqual(
            d.seq_node_idx[1], [10, 1, 2, 3, 4, 5, 6, 11, 12, 13]
        )
        self.assertListEqual(d.merge_node_idx, [1, 2, 3, 4, 5, 6])
        self.assertListEqual(d.seq_end_idx, [8, 12])
        # only substitution
        self.assertTupleEqual(d.to_POA(), ("GTACACGCTA", "GTACACGCGA"))

    def test_ex3(self):
        d = self._debruijn_w_basic_check('./tests/data/ex3.fasta')
        # caused by duplicated kmers
        self.assertListEqual(
            d.seq_node_idx[0], list(range(9))
        )
        self.assertListEqual(
            d.seq_node_idx[1], [9, 1, 2, 3, 4, 10, 11, 7, 12]
        )
        self.assertListEqual(d.merge_node_idx, [1, 2, 3, 4, 7])
        self.assertListEqual(d.seq_end_idx, [7, 7])
        # insertion/deletion
        self.assertTupleEqual(d.to_POA(), ("GTACACGTATG", "GTACACG-ATG"))

    def test_ex4(self):
        d = self._debruijn_w_basic_check('./tests/data/ex4.fasta')
        self.assertListEqual(
            d.seq_node_idx[0], list(range(10))
        )
        self.assertListEqual(
            d.seq_node_idx[1], [10, 1, 2, 3, 11, 6, 7, 8, 12]
        )
        self.assertListEqual(d.merge_node_idx, [1, 2, 3, 6, 7, 8])
        self.assertListEqual(d.seq_end_idx, [8, 8])
        # insertion/deletion, but the gap can be at either place
        self.assertIn(
            d.to_POA(), [
                ("GTACAAGCGA", "GTACA-GCGA"), ("GTACAAGCGA", "GTAC-AGCGA")
            ]
        )

    def test_ex5(self):
        d = self._debruijn_w_basic_check('./tests/data/ex5.fasta')
        self.assertListEqual(
            d.seq_node_idx[0], list(range(7))
        )
        self.assertListEqual(
            d.seq_node_idx[1], [7, 1, 2, 8, 9, 10]
        )
        self.assertListEqual(d.merge_node_idx, [1, 2])
        self.assertListEqual(d.seq_end_idx, [-1, -1])
        # insertion/deletion
        self.assertTupleEqual(d.to_POA(), ("ACGTAGACG", "ACGTA-ACG"))


if __name__ == '__main__':
    unittest.main()
