# DBGA: De Bruijn Graph Alignment Tool

**DBGA** is a Python library with a command line interface (CLI) designed for closely-related sequence alignment. It is capable of both global pairwise and multiple sequence alignment. DBGA uses De Bruijn graphs with existing alignment algorithms to boost both time and memory efficiency for sequence alignment. 

##  Getting Started

### Dependencies

DBGA installation requires Python 3.8 or newer. Additional libraries listed below are also required to run DBGA. They can be installed with `pip install -r requirements.txt`.

- [click](https://pypi.org/project/click/) - version 8.1.3
- [cogent3](https://pypi.org/project/cogent3/) - version 2022.8.24a1
- [graphviz](https://pypi.org/project/graphviz/) - version 0.20
- [nox](https://pypi.org/project/nox/) - version 2022.8.7
- [pytest](https://pypi.org/project/pytest/) - version 7.0.1

### Installing DBGA

Clone the DBGA repository to your machine.

```bash
git clone https://github.com/xingjianleng/DBGA.git
```

Change working directory to DBGA folder

```bash
cd DBGA/
```

Install Python `flit` package for installing DBGA

```bash
pip3 install flit
```

Use `flit` to install DBGA to current Python environment 

```bash
flit install -s
```

## Usage

To run the command line interface, typing `python3 src/dbga/cli.py` in the terminal. 

```
Usage: cli.py [OPTIONS]

Options:
  -i, --infile PATH       Input original sequences file  [required]
  -o, --outfile PATH      Output aligned sequences file  [required]
  -k INTEGER              The kmer size for constructing De Bruijn graph
                          [required]
  -m, --moltype TEXT      The input sequence molecular type  [required]
  --match INTEGER         score for two matching nucleotide (pairwise only)
  --transition INTEGER    cost for DNA transition mutation (pairwise only)
  --transversion INTEGER  cost for DNA transversion mutation (pairwise only)
  -d INTEGER              costs for opening a gap (pairwise only)
  -e INTEGER              costs for extending a gap (pairwise only)
  --model TEXT            The model for multiple sequence alignment (MSA only)
  --indel_rate FLOAT      One parameter for the progressive pair-HMM (MSA
                          only)
  --indel_length FLOAT    One parameter for the progressive pair-HMM (MSA
                          only)
  --help                  Show this message and exit.
```

Or, DBGA can be used directly as a Python package as shown in [Example Usage](#Example Usage) below.

## Example Usage

### Example1

Use the command line interface

```bash
python3 src/dbga/cli.py -i tests/data/substitution_middle.fasta -o out.fasta -k 3 -m dna
```

### Example2

Use directly as a Python package

```python
from dbga.debruijn_pairwise import DeBruijnPairwise


dbg = DeBruijnPairwise("tests/data/substitution_middle.fasta", k=3, moltype="dna")
dbg.alignment()
```

## Example Output

Output from above [Example1](#Example1)

```
Number of sequences: 2, running De Bruijn pairwise alignment
Alignment task finishes within 0.01 seconds!
Alignment results saved in path/to/out.fasta!
```

Output from above [Example2](#Example2)

```
seq1	GTACAAGCGA
seq2	.....C....
```
