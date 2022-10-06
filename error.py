import numpy as np

# compute mean absolute error (MAE)
def compute_mae(array):
    indices = np.where(array.T[2]>0)
    actual_vals = array.T[0][indices]
    predicted_vals = array.T[1][indices]
    return np.mean(np.abs(actual_vals - predicted_vals))


# compute root mean squared error (RMSE)
def compute_rmse(array):
    indices = np.where(array.T[2]>0)
    actual_vals = array.T[0][indices]
    predicted_vals = array.T[1][indices]
    return np.sqrt(np.mean((actual_vals - predicted_vals) ** 2))


# compute symmatric mean absolute percentage error (SMAPE)
# we compute "symmetric" percentage errors because it's more balanced for small
# values
def compute_smape(array):
    indices = np.where(array.T[2]>0)
    actual_vals = array.T[0][indices]
    predicted_vals = array.T[1][indices]
    errors = np.abs(actual_vals - predicted_vals)
    percentage_errors = errors / (np.abs(actual_vals) + np.abs(predicted_vals))
    return np.mean(percentage_errors)


# compute coefficient of determination (R^2)
def compute_r2(array):
    indices = np.where(array.T[2]>0)
    actual_vals = array.T[0][indices]
    predicted_vals = array.T[1][indices]
    se_line = np.sum((predicted_vals - actual_vals) ** 2)
    se_y = np.sum((actual_vals - np.mean(actual_vals)) ** 2)
    np.seterr(divide='ignore')
    return 1 - (se_line / se_y)
