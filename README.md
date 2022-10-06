# FIST-nD

Functional interpretation of spatial transcriptomics data usually requires non-trivial pre-processing steps and other accompany supporting data in the analysis due to the high sparsity and incompleteness of spatial RNA profiling, especially in 3D constructions. This software, FIST-nD, Fast Imputation of Spatially-resolved transcriptomes by graph-regularized Tensor completion in n-Dimensions imputes 3D as well as 2D spatial transcriptomics data. FIST-nD is implemented based on a novel graph-regularized tensor decomposition method, which imputes spatial gene expression data using high-order tensor structure and relations in spatial and gene functional graphs. The implementation, accelerated by GPU or multicore parallel computing, can efficiently impute high-density 3D spatial transcriptomics data within a few minutes.

## Installing

### Using pip

To install using [pip](https://pypi.org/project/fistnd/):

```
pip3 install fist-nd
```

### From Source

To install:

```
git clone https://github.com/kuanglab/FIST-nD
```

Dependencies can be found in the `environment.yml` file (for CPU) and `environmentGPU.yml` (for GPU).See https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html for instructions.


## Running

The basic structure of running the tool is as follows. If installed from pip:

```
python3 -m fistnd [input data] [output path] [optional arguments]
```

If installed from source:

```
python3 FIST.py [input data] [output path] [optional arguments]
```

A full walkthrough can be found [here](example/walkthrough.md).


## Modes and Necessary Arguments

Each mode has two necessary arguments, an input data, and a path to the output. **Note: For all modes, the output is specified by the path to the directory in which to place the output.** The table summarizes the three modes and their corresponding input data requirements and output details. The preprocessing, algorithm, and postprocessing modes can be specified using the `--preprocessing`, `--algorithm`, and  `--postprocessing` flags. If not specifying an option, all modes are run sequentially.

| Flags | Input | Output Result |
| ---- | ----- | ------ |
| None (Full Pipeline) | Count Matrix (specified below) | Imputed count matrix (`FIST_output.csv`) |
| Preprocessing | Count Matrix (specified below) | `.mat` file reprsenting tensor and PPI, `.json` file for metadata
| Algorithm | `.mat` file representing tensor and PPI, | `.mat` files reprsenting imputed tensor
| Postprocessing | Imputed tensor and metadata | Imputed ccunt matrix (`FIST_output.csv`) |

## Input Data Formatting

### Count Matrix

Data must be formatted as a `.csv` or `.parquet` file, where rows contain spots and columns contain genes. The first row must contain all the gene names in one of three formats: Entrex, Emsembl, or official gene symbol. The first column must contain the position of the spot, seperated by 'x' characters (ex. 10x20). The counts should be integers representing counts of the RNA molecules. An example input file is given in the file `example/example_data.csv`.

### Parquet File Format

FIST also understands the binary  `.parquet` file format, which is recommended for larger datasets for faster reading/writing. The `.parquet` file format can be substituted any place for `.csv` in the inputs, and the `--outformat` argument can be used to control the output.

### PPI Network (Recommended)

If specified, the PPI Network must be a tab-delimited file containing two columns named `Official Symbol Interactor A` and `Official Symbol Interactor B`. These columns should contain genes represented by official gene symbols, where two genes occuring in the same row indicates an interaction between their corresponding proteins. Networks in this format can be downloaded from BioGRID (https://downloads.thebiogrid.org/BioGRID/Release-Archive/BIOGRID-4.4.201/).

If not specified, the program will default to a PPI with an adjacency matrix equivalent to the identity matrix. This represents the belief that each gene interacts with itself, and no other genes.

## 10X Visium Data

To use data directly from 10X genomics Visium technology, there exists the `--visium` flag. In this case, the input will be specified as a directory with the following structure:

```
├── <dir>
    │   ├── filtered_feature_bc_matrix
    │   │   ├── barcodes.tsv.gz
    │   │   ├── features.tsv.gz
    │   │   └── matrix.mtx.gz 
    │   ├── spatial
    │   │   ├── tissue_positions_list.csv
    └── ...
```
## Binning

Because of the tensor decomposition, FIST can only impute spots that are arranged in a grid. **For 2D spatial transcriptome data, this is generally how the data is presented, and so the binning argument should be avoided.** However, any 3D data that has been transformed according to a reference atlas will require binning. In this case, the `--binning` (`-n`) argument should be used to determine the size of the final tensor. This argument can be provided in the format `XxYxZ`, where `X`, `Y`, and `Z` are integers corresponding to the spatial dimensions of the imputed tensor (ex. `15x15x15`).

## Full Optional Argument Description

Columns Pre, Algo, and Post indicate whether the argument is used in the preprocessing, algorithm, and  postprocessing modes.

| Argument | Short | Pre | Algo | Post | Description | Default |
| --------     | ---------- | --- | --- | --- | ----------- | ------- |
|`--verbose`   | None       | ✓ | ✓ | ✓ | Verbose output. | |
|`--visium`    | None       | ✓ |   |    | Described above.
|`--report`   | None       | | | ✓ | Generate PDF report of imputation. | |
|`--ppi`       | `-p`       | ✓ | | | A path to a protein-protein interaction network, specified in the format above. | Described above. |
|`--binning`|`-n`| ✓ | | | Described above. | None |
|`--nodiscrete`|None| ✓ | | | Disables automatic recognition and binning of discrete dimensions. Not recommended. | |
|`--rotate`|None| ✓ | | | Rotates data using PCA. | |
|`--geneformat`|`-g`| ✓ | | | Format of the gene names/symbols in the first row of the counts matrix. Must be one of `symbol`, `entrez`, or `ensembl`. | `symbol`
|`--organism`   |`-o`| ✓ | | | Used with `gene_format` to convert the provided gene format into official gene symbols. | `human`|
|`--geneplot`   |`-gp`| ✓ | | | Genes of interest to plot in final report, seperated by commas. Same format as columns of data file. | `None`|
|`--validation`|`-v`| ✓ |  | | Percentage of the data to hold out for validation. | `0`|
|`--lam`    |`-l`|    | ✓ | | The hyperparameter λ, as detailed in the paper. | `0.1`|
|`--rank`      |`-k`|    | ✓ | | The rank of the tensor to use. | `200`|
|`--stopcrit`  |`-s`|    | ✓ | | Stopping criteria, as a float. | `1e-4`|
|`--maxiters`  |`-i`|    | ✓ | | Maximum number of iterations to run FIST. | `500`|
|`--seed`      |`-r`|    | ✓ | | Random seed to use for validation for consistency. | None |
|`--backend`   |`-b`|    | ✓ | | For Python, the backend for the tensorly library to use. Can be `numpy` to run on CPU or `cuda` to run on GPU. | `numpy`|
|`--outformat`   |`-of`|    |  | ✓| Will output data in `.parquet` format if argument equals `parquet`, otherwise outputs in `.csv` format. | `csv`|
|`--metadata`   |`-m`|    | ✓ | ✓ | Metadata written to a file in preprocessing/algorithm steps. Required by algorithm step to add to metadata. | None |
