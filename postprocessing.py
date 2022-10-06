import numpy as np
import pandas as pd
import scipy.stats


def postprocessing(T_imputed, metadata, verbose=False):

    og_coords = metadata["coords"]
    n_dim = og_coords[0].count("x") + 1
    to_interp = np.array([coord.split("x") for coord in og_coords]).astype(float)

    if verbose:
        print("Rescaling coordinates...", flush=True)

    # convert original coordinates to tensor scale for interpolation
    for dim, data_range in zip(range(n_dim), metadata["data_ranges"]):
        if len(data_range) > 2:
            # this is the case in which the data has a discrete range
            # we replace using Pandas replacement function as it's easier than
            # using numpy
            pandas_col = pd.Series(to_interp[:, dim])
            pandas_col.replace(
                sorted(pandas_col.unique()),
                value=list(range(len(pandas_col.unique()))),
                inplace=True,
            )
            to_interp[:, dim] = np.abs(pandas_col.to_numpy() - 1e-7)
        else:
            # otherwise, we used the percentile bins

            # to unmap these, we need to do something a little different, as
            # we care about the spatial location inside the bins
            n_bins = T_imputed.shape[dim + 1]

            percentiles = np.array(
                [
                    scipy.stats.percentileofscore(to_interp[:, dim], i)
                    for i in to_interp[:, dim]
                ]
            )
            # the math is a bit weird here, because the bins at the start and
            # end are half as big as the rest for interpolation purposes bc
            # we have points at the edges of the data
            percentiles *= (n_bins - 1e-7) / 100
            bottom_bin = np.where(percentiles < 0.5)
            percentiles[bottom_bin] = percentiles[bottom_bin] * 2 - 0.5
            top_bin = np.where(percentiles > n_bins - 0.5)
            percentiles[top_bin] = (
                ((percentiles[top_bin] - (n_bins - 0.5)) * 2) + n_bins - 0.5
            )
            bins = (percentiles + 0.5).astype(int)

            # we need to get different cutoffs for interpolation
            quantile_cutoffs = list(
                np.linspace(0, 1, n_bins, endpoint=False) + (1 / (2 * n_bins))
            )
            quantile_cutoffs = np.array([0] + quantile_cutoffs + [1])
            bin_cutoffs = np.quantile(to_interp[:, dim], quantile_cutoffs)

            portion_of_bins = to_interp[:, dim] - bin_cutoffs[bins]
            portion_of_bins /= (1 + 1e-7) * (bin_cutoffs[bins + 1] - bin_cutoffs[bins])
            to_interp[:, dim] = bins + portion_of_bins

    # We do the multilinear interpolation algorithm ourselves because we can
    # get the coefficients, then appply to all genes. AFAIK Python has no good
    # way to get multilinear interpolation coefficients.

    if verbose:
        print("Finding interpolation coefficients...", flush=True)

    coefficients = []
    corners = np.array(np.where(np.zeros([2] * n_dim) == 0)).T
    for point in to_interp:
        # get the distances to the corner points
        proportions = point - np.floor(point)
        coeff = []
        for corner in corners:
            # get the size of the rectangle for each corner (the coefficient)
            coeff.append(
                np.prod(corner * proportions + (1 - corner) * (1 - proportions))
            )
        coefficients.append(coeff)
    # for every point, we have an array of coefficients for the 2^ndim corners
    coefficients = np.array(coefficients)

    if verbose:
        print("Computing interpolated expression at each spot...", flush=True)
    # now we compute the interpolated expression at each point
    expressions = []

    # here, we pad T_imputed so that each coordinate can represent the center of the bins
    T_for_interp = np.pad(T_imputed, [(0, 0)] + [(1, 1) for _ in range(n_dim)], "edge")

    for point, coeff in zip(to_interp, coefficients):
        point_corners = np.floor(point) + corners
        point_corners = point_corners.astype(int)
        corner_expressions = []
        # get expression at all corners
        for point_corner in point_corners:
            corner_expressions.append(
                T_for_interp[(slice(None),) + tuple(point_corner)]
            )
        # compute weighted average of corners and store as final expression
        expressions.append(np.sum(coeff * np.array(corner_expressions).T, axis=1))

    # as a bonus feature, we calculate highly-variable genes and report them
    vars = np.var(expressions, axis=0)
    vars /= np.mean(expressions, axis=0)
    variable_indices = vars.argsort()[-10:]
    variable_genes = []
    for index in variable_indices:
        variable_genes.append(metadata["gene_names"][index])
    metadata["variable_genes"] = variable_genes

    # last thing we do is undo the log transformation
    expressions = np.exp(np.array(expressions)) - 1
    expressions = expressions.astype(np.single)

    # save the data in a pandas dataframe and return
    df = pd.DataFrame(expressions, columns=metadata["gene_names"])
    df.index = og_coords
    df.index.name = "spot"

    # we store the genes to plot in the metadata
    for gene in metadata["gene_plot"]:
        if gene in df.columns:
            metadata[gene + "_post"] = df[gene].to_list()
        else:
            print(f"Warning: gene {gene} was dropped in preprocessing.")

    return df
