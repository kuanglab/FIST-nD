import tensorly as tl
from scipy.io import savemat, loadmat
import numpy as np
import sys
import os


def SpatialNN(T, test_subs = None, to_fill="all", verbose=False):
    """
    Implements nearest-neighbor imputation.

    Parameters
    ----------
    T : tl.tensor
        input tensor (dim: np-n1-n2-...)
    M : tl.tensor
        mask tensor (dim: np-n1-n2-... or n1-n2-...), 1 if value present,
        0 if value absent
    to_fill : tl.tensor, default None
        if provided, SpatialNN will fill only the provided indices, if not,
        all indices where M==0 will be filled
    verbose : bool, default False
        True to print progress updates

    Returns
    -------
    T : tl.tensor
        nearest-neighbor imputed tensor

    """

    T = tl.tensor(np.asfortranarray(T).copy())
    M = tl.zeros(T.shape)
    M[tl.where(T > 0)] = 1
    
    # for verbose output
    total = len(to_fill)
    last_update = 0
    total_updates = 100

    if test_subs is not None:
        # for verbose output
        total = len(test_subs)
        for i, test_sub in enumerate(test_subs):
            # find the spots that we can impute from
            train_subs = tl.tensor(tl.where(M[test_sub[0]] != 0)).T
            # compute this distance to all these spots
            dis = tl.sum((train_subs - test_sub[1:]) ** 2, axis=1)
            dis[tl.where(dis == 0)] = 1000
            # find the closest spot and impute it
            closest = train_subs[(dis == tl.min(dis))]
            T[tuple(test_sub)] = tl.mean(T[test_sub[0]][tuple(closest.T)])
            if verbose and int(total_updates * i / total) != last_update:
                print(
                    f"SpatialNN: {100*i/total:.2f}% complete ({i}/{total} spots imputed)",
                    flush=True,
                )
                last_update = int(total_updates * i / total)
        return np.ascontiguousarray(T)

    # if the user tells us, we impute only the spots with 0 expression across all genes
    if to_fill == "crossval":
        to_fill = tl.tensor(tl.where(tl.sum(T, axis=0) == 0)).T
    # else, if the user does not specify, we impute all the spots
    else:
        to_fill = tl.tensor(tl.where(tl.min(M, axis=0) == 0)).T

    total = len(to_fill)
    T = np.array(T)
    for i, test_sub in enumerate(to_fill):
        # get the slice of the spot that we're imputing
        vals_test = T[(slice(None),) + tuple(test_sub)]
        # get the genes that we need to impute
        nnz_id = np.where(vals_test == 0)[0]
        for val in nnz_id:
            # find the spots that we can impute from
            train_subs = np.array(np.where(M[val] != 0)).T
            # very edge case where no other spots to impute from
            # here we just guess 0
            if len(train_subs)==0:
                T[val][tuple(test_sub)] = 0
                continue
            # compute this distance to all these spots
            dis = np.sum((train_subs - test_sub) ** 2, axis=1)
            dis[np.where(dis == 0)] = 1000
            # find the closest spot and impute it
            closest = train_subs[(dis == np.min(dis))]
            T[val][tuple(test_sub)] = np.mean(T[val][tuple(closest.T)])
        if verbose and int(total_updates * i / total) != last_update:
            print(
                f"SpatialNN: {100*i/total:.2f}% complete ({i}/{total} spots imputed)",
                flush=True,
            )
            last_update = int(total_updates * i / total)
    return np.ascontiguousarray(T)


def main():
    # create the output path if need be
    if not (os.path.exists(sys.argv[2])):
        os.makedirs(sys.argv[2])

    data = loadmat(sys.argv[1])

    T = data["T"]
    if "test_subs" in data:
        test_subs = data["test_subs"]
        print(test_subs)
    else:
        test_subs = None
    T_imputed = SpatialNN(T, test_subs, to_fill="all", verbose=True)

    savemat(
        os.path.join(sys.argv[2], "imputedtensorSNN.mat"),
        {"T_imputed": T_imputed},
    )


if __name__ == "__main__":
    main()
