# DBGA: de Bruijn Graph Alignment Tool

**DBGA** is a Python library with a command line interface (CLI) designed for closely-related sequence alignment. It is capable of both global pairwise and multiple sequence alignment. DBGA uses de Bruijn graphs with existing alignment algorithms to boost both time and memory efficiency for sequence alignment. 

##  Getting Started

### Dependencies

DBGA installation requires Python 3.8 or newer. Additional libraries listed below were used when implementing the DBGA library.

- [click](https://pypi.org/project/click/)
- [cogent3](https://pypi.org/project/cogent3/)
- [graphviz](https://pypi.org/project/graphviz/)
- [nox](https://pypi.org/project/nox/)
- [pytest](https://pypi.org/project/pytest/)

### Installing DBGA

Clone the DBGA repository to your machine.

```bash
git clone https://github.com/xingjianleng/DBGA.git
```

Change working directory to DBGA folder

```bash
cd DBGA/
```

#### User install

Install the dbga package using `pip3`

```
pip3 install .
```

#### Developer install

Install Python `flit` package for installing DBGA

```
pip3 install flit
```

Use `flit` to install DBGA to current Python environment 

```bash
flit install -s
```


## Usage

To run the command line interface, typing `dbga --help` in the terminal. 

```
Usage: dbga [OPTIONS]

Options:
  -i, --infile PATH       Input original sequences file  [required]
  -o, --outfile PATH      Output aligned sequences file  [required]
  -k INTEGER              The kmer size for constructing de Bruijn graph
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

Or, DBGA can be used directly as a Python package shown in [Example Usage](#Example Usage) below.

## Example Usage

### Example1

Use the command line interface

```
dbga -i tests/data/substitution_middle.fasta -o out.fasta -k 3 -m dna
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

```bash
Number of sequences: 2, running de Bruijn pairwise alignment
Alignment task finishes within 0.01 seconds!
Alignment results saved in path/to/out.fasta!

cat out.fasta
>seq1
GTACAAGCGA
>seq2
GTACACGCGA
```

Output from above [Example2](#Example2)

```
seq1	GTACAAGCGA
seq2	.....C....
```

## Run tests

To run tests in the repository, simply install and run [nox](https://pypi.org/project/nox/) with the following commands to test across Python 3.8 to 3.10.

```bash
pip install nox
nox
```

**Note:** The code in `datasampler.py` and `kmersize.py` were not tested as they were not used in the algorithm implementation.

## Benchmark

The benchmark results and related datasets can be found at [Cloudstor](https://cloudstor.aarnet.edu.au/plus/s/l5k2v2rzS5axfRD).
