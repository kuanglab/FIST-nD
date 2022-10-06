import numpy as np
import pandas as pd
import mygene
import csv
import gzip
import os
import scipy.io
import tensorly as tl
import warnings
import scipy.stats
from sklearn.decomposition import PCA


def preprocessing(
    input_path: str,
    PPI_dir: str,
    visium: bool,
    binning,
    nodiscrete=False,
    validation=None,
    gene_format="symbol",
    species="human",
    geneplot="",
    rotate=False,
    verbose=False,
):
    """
    Converts a csv matrix of gene counts into an expression tensor and a PPI graph.

    Parameters
    ----------
    input_path : str
        Path to the input .csv file or visium directory.
    PPI_dir : str
        Path to the PPI in BioGRID format (see https://downloads.thebiogrid.org/BioGRID/Release-Archive/BIOGRID-4.4.201/)
    visium : bool
        whether data is formatted in the 10X Visium format
    binning : tuple
        output shape of the tensor, without the first dimension (the gene dim)
    output_dir : str, default None
        Path for the output MATLAB files, if None, files are not saved.
    gene_format : str, default "symbol"
        format of the gene names in first row, "symbol","entrez", or "ensembl"
    species : str, default "human"
        species the data belongs to, for converting of gene symbols

    Returns
    -------
    T : np.array
        binned tensor of shape (n_genes, dimensions)
    W_G : np.array
        PPI network as adjacency matrix of shape (n_genes, n_genes)
    metadata : dict
        data needed to reconstruct original data format
    """

    # create dictionary to hold metadata values
    metadata = dict()

    if verbose:
        print("Reading input data...", flush=True)

    if visium:
        data = visium_data_prep(input_path)
    else:
        # we read the first row of the df to get the data types of each column
        # this reduces the amount of memory needed by the "real" read.csv call

        # we also accept .parquet format
        if input_path.endswith("parquet"):
            data = pd.read_parquet(input_path)
        else:
            data = pd.read_csv(input_path, nrows=1)
            dtypes = data.dtypes.to_dict()
            data = pd.read_csv(input_path, dtype=dtypes)
        
        # check if first column is strings
        if data[data.columns[0]].dtype!='O':
            data.reset_index(inplace=True)

    # get count of original genes to save for later
    og_genes = len(data.columns) - 1

    # get the dimensionality of the data
    n_dim = data.loc[0][0].count("x") + 1
    if binning is not None:
        dimensions = binning.lower().split("x")
        dimensions = tuple([int(dim) for dim in dimensions])

        if n_dim != len(dimensions):
            raise ValueError("bad input dimensions")
    if binning == "" and n_dim == 3:
        dimensions = [15, 15, 15]
        warnings.warn(
            "No binning size specified. Defaulting to 15x15x15.",
        )

    dim_names = [f"x{i+1}" for i in range(n_dim)]
    spot_col = data.columns[0]

    # save original coordinates for metadata
    og_coords = data[spot_col]

    # split spot column into n columns
    data[dim_names] = data[spot_col].str.split("x", expand=True)
    data = data.drop(columns=[spot_col])

    if verbose and binning is not None:
        print("Binning and filtering coordinates...", flush=True)
    if verbose and binning is None:
        print("Filtering coordinates...", flush=True)

    data_ranges = []

    if rotate:
        print("Rotating dataset using PCA...")
        X = np.array([data[dim] for dim in dim_names[:-1]]).T
        pca = PCA()
        X_new = pca.fit_transform(X)
        for i, dim in enumerate(dim_names[:-1]):
            data[dim] = X_new.T[i]

    # if needed, convert the coordinates to a discrete scale the size of the bins
    new_dims = []
    if binning is not None:
        for n_spot, dim in zip(dimensions, dim_names):
            data[dim] = data[dim].astype(float)
            # we check for the user to see if the data is actually discrete
            # in this case, binning can mess it up, so we bin in a different way
            if len(data[dim].unique()) > 4 * n_spot or nodiscrete:
                # if the data appears continuous, we sort the data into bins
                # based on quantiles
                if len(data[dim].unique()) <= 4 * n_spot:
                    warnings.warn(
                        f"dim {dim[-1]} appears to be discrete.",
                    )
                data_ranges.append([min(data[dim]), max(data[dim])])
                data[dim] = np.array(
                    [scipy.stats.percentileofscore(data[dim], i) for i in data[dim]]
                )
                data[dim] *= (n_spot - 1e-7) / 100
                data[dim] = data[dim].astype(int)
                new_dims.append(n_spot)
            else:
                print(f"dim {dim[-1]} appears to be discrete. Converting to discrete.")
                unique_num = len(data[dim].unique())
                data_ranges.append(sorted(data[dim].unique()))
                data[dim].replace(
                    sorted(data[dim].unique()),
                    value=list(range(unique_num)),
                    inplace=True,
                )
                new_dims.append(unique_num)
        # we now update the dimension to reflect any dimensions that were made
        # discrete
        dimensions = tuple(new_dims)
    else:  # if not, just convert to ints
        dimensions = []
        for dim in dim_names:
            try:
                data[dim] = data[dim].astype(int)
            except ValueError:
                raise ValueError(
                    "Could not convert coordinates to integers. Check the binning (-n) argument."
                )
            unique_num = len(data[dim].unique())
            data_ranges.append([int(i) for i in sorted(data[dim].unique())])
            data[dim].replace(
                sorted(data[dim].unique()),
                value=list(range(unique_num)),
                inplace=True,
            )
            new_dims.append(unique_num)
            dimensions.append(max(data[dim]) + 1)
        dimensions = tuple(dimensions)

    # next, create a unique id for all genes to group by to collapse duplicate spots
    data["id"] = sum(
        [
            data[dim] * (n_spot ** i)
            for i, (n_spot, dim) in enumerate(zip(dimensions, dim_names))
        ]
    )

    data_sum = data.groupby(["id"]).sum()
    data_mean = data.groupby(["id"]).mean()

    # here, we remove all entries (bins) that have a count less than three
    pd.set_option("mode.chained_assignment", None)  # remove a warning
    invalid = data_sum.iloc[:, :-n_dim] < 3
    data_mean.iloc[:, :-n_dim][invalid] = 0

    # doing this made some genes have zero expression, we filter those out
    data_mean = data_mean.loc[:, (data_mean.sum(axis=0) != 0)]

    # we fiter out genes that have less than 8 counts total
    num_nonzero = np.sum(data_mean.gt(0), axis=0)
    data_mean = data_mean.loc[:, (num_nonzero > 8)]

    original_names = list(data_mean.columns[:-n_dim])

    if PPI_dir is not None:
        if verbose:
            print("Reading PPI...", flush=True)
        ppi_BioGRID = read_PPI(PPI_dir)

        if verbose and gene_format != "symbol":
            print("Converting genes to official symbols...", flush=True)

        # we have to convert genes from the format they were given to official gene symbol
        gene_names = list(data_mean.columns)[:-n_dim]
        if gene_format != "symbol":
            gene_symbols = mygene.MyGeneInfo().querymany(
                gene_names,
                scopes=f"{gene_format}.gene",
                verbose=False,
                species=species,
            )

            gene_symbols = [
                gene["symbol"] if "symbol" in gene.keys() else None
                for gene in gene_symbols
            ]
        else:
            gene_symbols = gene_names

        # filter the PPI based on which genes we actually measured
        ppi_BioGRID = ppi_BioGRID[
            ppi_BioGRID["Official Symbol Interactor A"].isin(gene_symbols)
        ]
        ppi_BioGRID = ppi_BioGRID[
            ppi_BioGRID["Official Symbol Interactor B"].isin(gene_symbols)
        ]

        # only keep genes that were properly converted
        valid_gene = [True if symbol is not None else False for symbol in gene_symbols]

        # save the original column names for later
        original_names = list(np.array(data_mean.columns[:-n_dim])[valid_gene])

        # rename the columns to the standard gene symbols
        data_mean = data_mean.loc[:, valid_gene + [True] * n_dim]
        data_mean.columns = [
            symbol for symbol in gene_symbols if symbol is not None
        ] + dim_names

        if len(ppi_BioGRID) < (len(data_mean.columns) // 4):
            warnings.warn(
                "Abnormally low number of PPI connections used. Ensure the correct PPI and gene symbols were used."
            )

    else:
        ppi_BioGRID = pd.DataFrame()

    n_genes = len(data_mean.columns) - n_dim
    genes = list(data_mean.columns[:-n_dim])

    if verbose:
        print(f"Genes kept: {n_genes} of {og_genes} ({100*n_genes/og_genes:.1f}%)")
        print(
            f"PPI connections used in model: {len(ppi_BioGRID)} ({len(ppi_BioGRID) / n_genes:.1f} connections per gene)"
        )
        print(f"Creating tensor of shape {(n_genes,) + dimensions}...")

    # store results in metadata for later
    metadata["og_genes"] = og_genes
    metadata["n_genes"] = n_genes

    T = create_tensor_from_dataframe(data_mean, dimensions)

    W_g = create_graph_from_bioGRID(ppi_BioGRID, genes)

    metadata["data_ranges"] = data_ranges
    metadata["gene_names"] = original_names
    metadata["coords"] = og_coords.to_list()
    metadata["shape"] = str(T.shape)
    metadata["val_spots"] = None
    if geneplot == "":
        metadata["gene_plot"] = []
    else:
        metadata["gene_plot"] = geneplot.split(",")
    for gene in metadata["gene_plot"]:
        if gene in original_names:
            metadata[gene + "_pre"] = data[gene].to_list()
        else:
            warnings.warn(f"gene {gene} was dropped in preprocessing")

    # before returning the final tensor, we do a check to make sure that there
    # are no tensor slices with zero points. when this occurs, it can mess up
    # the model
    for dim in range(1, 1 + n_dim):
        to_sum = list(range(n_dim + 1))
        to_sum.remove(dim)
        if np.any(np.trim_zeros(np.sum(T, axis=tuple(to_sum))) == 0):
            warnings.warn(
                f"Slice with zero expression detected on dim {dim}. Consider lowering the binning dimensions."
            )

    # finally, we take out a validation set from the tensor if requested
    nonzero = tl.tensor(tl.where(T != 0)).T

    if validation is not None:
        if verbose:
            print("Performing validation split...", flush=True)
        test_subs = nonzero[
            np.random.choice(
                np.arange(len(nonzero)),
                size=int(len(nonzero) * validation),
                replace=False,
            )
        ]
        vals_test = T[tuple(test_subs.T)]

        T[tuple(test_subs.T)] = 0

        metadata["val_spots"] = len(vals_test)

        return T, W_g, metadata, test_subs, vals_test

    return T, W_g, metadata, nonzero, T[tuple(nonzero.T)]


def visium_data_prep(input_path):
    # Visium data prep adapted from official 10X genomic site
    # https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/matrices
    mat = scipy.io.mmread(
        os.path.join(input_path, "filtered_feature_bc_matrix/matrix.mtx.gz")
    )

    features_path = os.path.join(
        input_path, "filtered_feature_bc_matrix/features.tsv.gz"
    )
    gene_names = [
        row[1] for row in csv.reader(gzip.open(features_path, "rt"), delimiter="\t")
    ]
    barcodes_path = os.path.join(
        input_path, "filtered_feature_bc_matrix/barcodes.tsv.gz"
    )
    barcodes = [
        row[0] for row in csv.reader(gzip.open(barcodes_path, "rt"), delimiter="\t")
    ]

    spots = pd.read_csv(
        os.path.join(input_path, "spatial/tissue_positions_list.csv"), header=None
    )

    # we int-divide y-spots by 2 to properly align the spots
    coords = spots[2].astype(str) + "x" + (spots[3] // 2).astype(str)
    spot_barcodes = list(spots[0])
    writable_coords = []

    for code in barcodes:
        writable_coords.append(coords[spot_barcodes.index(code)])

    data = pd.DataFrame(mat.toarray().T)
    data.columns = gene_names
    data.insert(0, column="spot", value=writable_coords)
    return data


def read_PPI(PPI_dir):
    # we just say that those columns are strings to avoid pandas complaining
    dtypes = {
        "Entrez Gene Interactor A": str,
        "Entrez Gene Interactor B": str,
        "Score": str,
    }
    ppi_BioGRID = pd.read_csv(PPI_dir, sep="\t", dtype=dtypes)
    # we only use the official gene symbol
    ppi_BioGRID = ppi_BioGRID[
        ["Official Symbol Interactor A", "Official Symbol Interactor B"]
    ]
    return ppi_BioGRID


def create_tensor_from_dataframe(df, dimensions):
    # create empty tensor
    T = np.zeros((len(df.columns) - len(dimensions),) + dimensions)

    # for each spot, fill the tensor expression for that gene
    # we also do our log transform here
    for _, row in df.iterrows():
        row = np.array(row)
        spot = row[: -len(dimensions)]
        pos = row[-len(dimensions) :].astype(int)
        T[(slice(None),) + tuple(pos)] = np.log(spot + 1)

    return T


def create_graph_from_bioGRID(ppi_BioGRID, genes):
    # create the PPI network
    W_g = np.identity(len(genes))

    gene_dict = {gene: i for i, gene in enumerate(genes)}

    # for each interaction in the PPI, add it to the network
    for _, row in ppi_BioGRID.iterrows():
        index_A = gene_dict[row[0]]
        index_B = gene_dict[row[1]]
        W_g[index_A][index_B] = 1
        W_g[index_B][index_A] = 1

    return W_g
