import tensorly as tl
import numpy as np
from datetime import datetime
import functools
import error

def FIST(
    T,
    W_g,
    l=0.1,
    rank_k=None,
    stop_crit=1e-3,
    max_iters=500,
    val_indices=None,
    val_values=None,
    verbose=False,
    metadata=None,
    backend="numpy",
):
    """
    Implements the FIST algorithm.

    Parameters
    ----------
    T : tl.tensor
         input tensor (dim: np-n1-n2-...)
    W_g : tl.tensor
         PPI network in adjacency matrix form (dim: np-np)
    l : float, default 0.1
    rank_k : int, default None
         rank of the tensor in CPD form. if None, will guess based on PCA
    lambda : int, default 0.1
         value of hyperparameter lambda
    stop_crit : float, default 1e-3
         stopping criteria for FIST
    max_iters : int, default 500
         maximum number of iterations to run
    val_indices : tl.tensor, default None
         indices of validation data, if None not used (dim: numindices-ndim)
    val_values : tl.tensor, default None
         values corresponding to validation indices (dim: nimdices)

    Returns
    -------
    A : list of tl.tensors
        list of the CPD factor matrices learned by FIST

    """

    if backend == "cuda":
        tl.set_backend("pytorch")
    T = tl.tensor(T)
    M = tl.zeros(T.shape)
    M[tl.where(T > 0)] = 1
    n = T.shape  # tensor dimensions
    net_num = len(n)

    # construct spatial chain graphs
    W = [tl.tensor(W_g)]
    for dim in T.shape[1:]:
        graph = tl.diag(tl.ones(dim - 1), -1) + tl.diag(tl.ones(dim - 1), 1)
        W.append(graph)

    validation = False
    if val_indices is not None and val_values is not None:
        if val_indices.shape[0] != val_values.shape[0]:
            raise ValueError(
                f"Validation indices and values must have same first dimension."
            )
        validation = True
        prev_mae = 1e9
        prev_smape = 1e9
        prev_rmse = 1e9

        metadata["val_mae"] = []
        metadata["val_smape"] = []
        metadata["val_rmse"] = []

        val_values = tl.tensor(val_values)

    # graph normalization
    D = []
    for netid in range(net_num):
        # convert the graph to floats for normalization
        if backend == "numpy":
            W[netid] = tl.tensor(W[netid].astype("float64"))
        W[netid] = W[netid] - tl.diag(tl.diag(W[netid]))
        d = tl.sum(W[netid], 1)
        nonzero = tl.where(d != 0)
        d[nonzero] = d[nonzero] ** (-0.5)
        d = tl.tensor(np.expand_dims(d, axis=1))

        W[netid] = W[netid] * d
        W[netid] = d.T * W[netid]
        D.append(tl.diag(tl.sum(W[netid], 1)))

    A = []  # tensor in CPD form
    phi = []  # A^T dot A
    WA = []  # W dot A
    DA = []  # D dot A
    theta_W = []  # A^T dot W dot A
    theta_D = []  # A^T dot D dot A

    for netid in range(net_num):
        A.append(tl.tensor(np.random.uniform(size=(n[netid], rank_k))))
        phi.append(A[netid].T @ A[netid])
        WA.append(W[netid] @ A[netid])
        DA.append(tl.tensor(np.expand_dims(tl.diag(D[netid]), axis=1)) * A[netid])
        theta_W.append(A[netid].T @ WA[netid])
        theta_D.append(A[netid].T @ DA[netid])

    if backend == "cuda":
        T = T.to("cuda")
        M = M.to("cuda")
        to_convert = [W, A, phi, WA, DA, theta_W, theta_D]
        for set in to_convert:
            for i in range(len(set)):
                set[i] = set[i].to("cuda")
        val_values = val_values.to("cuda")

    # we save the time for each iteration to display in the metadata
    start = datetime.now()

    for iter in range(max_iters):
        A_old = A.copy()
        for i in range(net_num):
            # compute two MTTKRP operations that have the same factor
            T_unfolded = tl.unfold(T, i)
            kr_factors = tl.cp_tensor.khatri_rao(A, skip_matrix=i)
            num = T_unfolded @ kr_factors
            # this is the most memory intensive operation, so we strategically
            # order our operations and delete T_unfolded to get a small improvement
            del T_unfolded
            Y_hat = tl.cp_to_tensor((None, A)) * M
            Y_hat_unfolded = tl.unfold(Y_hat, i)
            denom = Y_hat_unfolded @ kr_factors
            del Y_hat_unfolded
            del Y_hat
            del kr_factors

            # calculate dJ2dAi-
            prod_phi_ks = []
            for j in range(net_num):
                phi_ks = [phi[k] for k in range(net_num) if k != i and k != j]
                if backend == "cuda":
                    prod = functools.reduce(hadamard, phi_ks)
                    prod_phi_ks.append(prod)
                else:
                    prod_phi_ks.append(tl.prod(phi_ks, axis=0))

            to_multiply = tl.zeros(phi[0].shape)
            if backend == "cuda":
                to_multiply = to_multiply.to("cuda")

            for j in range(net_num):
                if j != i:
                    to_multiply += theta_W[j] * prod_phi_ks[j]
            num_kronsum = A[i] @ to_multiply

            to_hadamard = [phi[j] for j in range(net_num) if j != i]
            if backend == "cuda":
                hadamard_product = functools.reduce(hadamard, to_hadamard)
            else:
                hadamard_product = tl.prod(to_hadamard, axis=0)
            num_kronsum += WA[i] @ hadamard_product

            # calculate dJ2dAi+, very similar to above
            to_multiply = tl.zeros(phi[0].shape)
            if backend == "cuda":
                to_multiply = to_multiply.to("cuda")
            for j in range(net_num):
                if j != i:
                    to_multiply += theta_D[j] * prod_phi_ks[j]
            denom_kronsum = A[i] @ to_multiply

            denom_kronsum += DA[i] @ hadamard_product

            # add together the two parts of the derivative and update Ai
            num = num + l * num_kronsum + 1e-10
            denom = denom + l * denom_kronsum + 1e-10
            A[i] = A[i] * (num / denom)

            # update vairables for next go-around
            phi[i] = A[i].T @ A[i]
            WA[i] = W[i] @ A[i]
            partial = tl.tensor(np.expand_dims(tl.diag(D[i]), axis=1))
            if backend == "cuda":
                partial = partial.to("cuda")
            DA[i] = partial * A[i]
            theta_W[i] = A[i].T @ WA[i]
            theta_D[i] = A[i].T @ DA[i]

        res = compute_res(A, A_old)
        if validation:
            mae, smape, rmse = compute_metrics(A, val_indices, val_values)
            metadata["val_mae"].append(float(mae))
            metadata["val_smape"].append(float(smape))
            metadata["val_rmse"].append(float(rmse))
            if verbose:
                print(
                    f"FIST Iter {iter} \tRes: {res:.5f}\tMAE: {mae:.5f}\tSMAPE: {smape:.5f}\tRMSE: {rmse:.5f}",
                    flush=True,
                )
            # we stop if all validation metrics are going in the wrong direction
            if mae > prev_mae and smape > prev_smape and rmse > prev_rmse:
                break
            prev_mae = mae
            prev_smape = smape
            prev_rmse = rmse
        else:
            if verbose:
                print(f"FIST Iter {iter}\tRes: {res}", flush=True)
        if res < stop_crit:
            break
    end = datetime.now()
    metadata["time_per_iter"] = str((end - start) / (iter))[2:-3]
    T_imputed = tl.cp_to_tensor((None, A))
    if backend == "cuda":
        T_imputed = np.array(T_imputed.cpu())
    return T_imputed


def compute_res(A, A_old):
    """Compute sum of percent squared difference between matrices."""
    res_num = 0
    res_denom = 0
    for i in range(len(A)):
        res_num += tl.sum((A[i] - A_old[i]) ** 2)
        res_denom += tl.sum(A_old[i] ** 2)
    return tl.sqrt(res_num / res_denom)


def compute_metrics(A, val_indices, val_values):
    # Compute MAE/MAPE/Rsq on validation data.
    T = tl.cp_to_tensor((None, A))
    predicted = T[tuple(val_indices.T)]
    good_vals = tl.where(val_values > 0)
    actual_vals = val_values[good_vals]
    predicted_vals = predicted[good_vals]
    # if the tensors are on the GPU, we port them to the CPU
    try:
        actual_vals = np.array(actual_vals.cpu())
        predicted_vals = np.array(predicted_vals.cpu())
    except:
        pass
    array = np.array([actual_vals, predicted_vals, np.ones(len(actual_vals))])
    mae = error.compute_mae(array)
    smape = error.compute_smape(array)
    rmse = error.compute_rmse(array)
    return mae, smape, rmse

# this is just for readibility
def hadamard(a, b):
    return a * b
