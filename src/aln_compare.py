from adaptive_debruijn import adpt_dbg_alignment
from debruijn_pairwise import deBruijn, to_alignment
from pathlib import Path


def compare(file_path):
    dbg = deBruijn(file_path, k=3, moltype="dna")
    aln_conventional = to_alignment(dbg)
    aln_new = adpt_dbg_alignment(file_path, thresh=2)
    return aln_conventional == aln_new


data_path = Path("../tests/data/").expanduser().absolute()
correct, wrong = [], []

for file_name in data_path.iterdir():
    if compare(file_name):
        correct.append(file_name.stem)
    else:
        wrong.append(file_name.stem)

print(f"Total {len(correct)} correct alignments. File names are {correct}")
print(f"Total {len(wrong)} wrong alignments. File names are {wrong}")
